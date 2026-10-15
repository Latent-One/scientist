//! `scientist chat`: the agent loop with you in it.
//!
//! You type a message      -> UserMessage event -> planner step(s)
//! You type /command       -> the same parser as the shell; runs as a manual action
//!
//! One planner step:
//!   propose (vLLM, schema-constrained) -> validate against state -> PlannerProposed event
//!   invalid? the errors go back to the planner (max 3 tries)
//!   respond / ask_human?  print and hand control back to you
//!   needs approval at this autonomy level?  ask you: y / n / free text (redirect) / /command
//!   execute -> RUNTIME result goes into the history the planner sees next step

use crate::actions::{self, Action, Ui};
use crate::cli::{self, Cli};
use crate::events::{Actor, Event};
use crate::learn;
use crate::planner::{self, PlannerOutput};
use crate::tools::Runtime;
use anyhow::Result;
use clap::Parser;
use rustyline::error::ReadlineError;
use rustyline::{Cmd as Key, ConditionalEventHandler, DefaultEditor, Event as KeyEv, EventContext, EventHandler, KeyCode, KeyEvent, Modifiers, RepeatCount};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// Shift+Tab: flip auto-approve, remember what was typed, and end this readline so the
/// prompt is redrawn with the new mode; the typed text is restored on the next prompt.
struct ToggleAuto {
    auto: Arc<AtomicBool>,
    toggled: Arc<AtomicBool>,
    saved: Arc<Mutex<String>>,
}

/// Ctrl-C at the prompt: clears typed text; on an empty line it arms "press again to exit".
struct CtrlC {
    empty: Arc<AtomicBool>,
}

impl ConditionalEventHandler for CtrlC {
    fn handle(&self, _: &KeyEv, _: RepeatCount, _: bool, ctx: &EventContext) -> Option<Key> {
        self.empty.store(ctx.line().is_empty(), Ordering::SeqCst);
        Some(Key::Interrupt)
    }
}

impl ConditionalEventHandler for ToggleAuto {
    fn handle(&self, _: &KeyEv, _: RepeatCount, _: bool, ctx: &EventContext) -> Option<Key> {
        self.auto.fetch_xor(true, Ordering::SeqCst);
        self.toggled.store(true, Ordering::SeqCst);
        *self.saved.lock().unwrap() = ctx.line().to_string();
        Some(Key::Interrupt)
    }
}

const HELP: &str = "\
Type a question or goal in plain language and the planner proposes one step at a time.
Type /<command> to act directly; it is the same as `scientist <command>` in the shell:
  /status   /log   /inspect c3   /evidence c3   /compare c3 c5   /workers
  /build --prototype diamond --formula C      /relax c1 --with mace --learn
  /mutate c1 --substitute 0:N                 /sweep c1 --from -3 --to 3 --points 7
  /generate --with mattergen --system B-C-N -n 2     /hull --with mattersim
  /search define --strategy evolutionary --objective min:energy_per_atom --pool c1
  /note <text> --refs c3     /level 2     /learn relaxation     /notebook
  /new <what you want to investigate>   start another experiment (planner drafts it, you review)
  Shift+Tab or /auto   toggle auto-approve for this session (validation and budgets still apply)
/help for this, /quit to leave.
Ctrl-C while the agent works: interrupt at once (finished results are kept). At the prompt: clear the line;
on an empty line press Ctrl-C twice to exit.";

/// How the chat loop begins.
pub enum Start {
    Idle,
    /// Your first message, sent to the planner right away.
    Message(String),
    /// The experiment was just created from your words: the planner starts on them.
    Begin,
}

/// `scientist` with no subcommand: the Claude-Code-style entry point. The working directory
/// selects the experiment; with none, your first message becomes the experiment.
pub fn session(rt: &mut Runtime, exp: Option<String>, first: Option<String>, learn_mode: bool) -> Result<()> {
    let exp = match exp { Some(e) => Some(e), None => cli::dir_experiment(rt)? };
    banner(rt, exp.as_deref())?;
    match exp {
        Some(e) => chat(rt, &e, learn_mode, first.map(Start::Message).unwrap_or(Start::Idle)),
        None => {
            let desc = match first {
                Some(f) => f,
                None => {
                    println!("  No experiment in this directory yet. Tell me what you want to investigate.");
                    println!("  (or /list to see experiments, /use <name> to open one here, /quit)\n");
                    let mut ed = DefaultEditor::new()?;
                    loop {
                        let line = match ed.readline("> ") {
                            Ok(l) => l.trim().to_string(),
                            Err(_) => return Ok(()),
                        };
                        match line.as_str() {
                            "" => continue,
                            "/quit" | "/exit" | "/q" => return Ok(()),
                            "/list" => {
                                for e in rt.store.experiments()? {
                                    println!("  {:<24} {:>5} events", e.name, e.events);
                                }
                            }
                            l if l.starts_with("/use ") => {
                                let name = l[5..].trim().to_string();
                                if rt.store.experiment_exists(&name)? {
                                    cli::select(rt, &name)?;
                                    return chat(rt, &name, learn_mode, Start::Idle);
                                }
                                println!("  no experiment {name}");
                            }
                            l if l.starts_with('/') => println!("  first describe what you want to investigate (or /list, /use <name>, /quit)"),
                            l => break l.to_string(),
                        }
                    }
                }
            };
            match cli::create_from_description(rt, &desc, false)? {
                Some(name) => chat(rt, &name, learn_mode, Start::Begin),
                None => Ok(()),
            }
        }
    }
}

fn banner(rt: &Runtime, exp: Option<&str>) -> Result<()> {
    let dir = std::env::current_dir().map(|p| p.display().to_string()).unwrap_or_default();
    // Ask the server what it is actually serving (the configured name is only an alias).
    let served: Option<String> = ureq::get(&format!("{}/v1/models", rt.cfg.planner.url))
        .timeout(std::time::Duration::from_millis(800))
        .call()
        .ok()
        .and_then(|r| r.into_json::<serde_json::Value>().ok())
        .and_then(|v| v["data"][0]["root"].as_str().map(String::from));
    let exp_line = match exp {
        Some(e) => {
            let st = rt.state(e)?;
            format!("{e} · {} candidates · L{} {}", st.candidates.len(), st.autonomy, cli::level_text(st.autonomy).split(':').next().unwrap_or(""))
        }
        None => "none yet (your first message starts one)".into(),
    };
    let rows = [
        format!("scientist {}  ·  local materials & chemistry instrument", env!("CARGO_PKG_VERSION")),
        format!("dir         {dir}"),
        format!("experiment  {exp_line}"),
        match &served {
            Some(m) => format!("planner     {m} · ready"),
            None => format!("planner     NOT REACHABLE at {} (scripts/serve_planner.sh); /commands still work", rt.cfg.planner.url),
        },
    ];
    let w = rows.iter().map(|r| r.chars().count()).max().unwrap_or(40) + 2;
    println!("╭{}╮", "─".repeat(w));
    for r in &rows {
        println!("│ {r}{} │", " ".repeat(w - 2 - r.chars().count()));
    }
    println!("╰{}╯", "─".repeat(w));
    println!("  type what you need · /help · Shift+Tab auto-approve · Ctrl-C interrupt\n");
    Ok(())
}

pub fn chat(rt: &mut Runtime, exp: &str, learn_mode: bool, start: Start) -> Result<()> {
    let mut exp = exp.to_string();
    rt.open_for_write(&exp)?;
    let mut rl = DefaultEditor::new()?;
    let hist = rt.cfg.data_dir().join("chat_history.txt");
    let _ = rl.load_history(&hist);
    let auto = Arc::new(AtomicBool::new(false));
    let toggled = Arc::new(AtomicBool::new(false));
    let saved = Arc::new(Mutex::new(String::new()));
    rl.bind_sequence(
        KeyEvent(KeyCode::BackTab, Modifiers::NONE),
        EventHandler::Conditional(Box::new(ToggleAuto { auto: auto.clone(), toggled: toggled.clone(), saved: saved.clone() })),
    );
    let empty_ctrl_c = Arc::new(AtomicBool::new(false));
    rl.bind_sequence(KeyEvent::ctrl('C'), EventHandler::Conditional(Box::new(CtrlC { empty: empty_ctrl_c.clone() })));
    let mut armed_exit: Option<std::time::Instant> = None;
    let first = match start {
        Start::Idle => None,
        Start::Message(m) => Some(Some(m)),
        Start::Begin => Some(None),
    };
    if let Some(msg) = first {
        let mut a = auto.load(Ordering::SeqCst);
        if let Err(e) = agent_turn(rt, &exp, msg, learn_mode, &mut a) {
            println!("error: {e:#}");
        }
        auto.store(a, Ordering::SeqCst);
    }
    loop {
        rt.clear_interrupt();
        let level = rt.state(&exp)?.autonomy;
        let on = auto.load(Ordering::SeqCst);
        let prompt = format!("[{exp} L{level}{}] > ", if on { " AUTO" } else { "" });
        let initial = std::mem::take(&mut *saved.lock().unwrap());
        let line = match rl.readline_with_initial(&prompt, (&initial, "")) {
            Ok(l) => l,
            Err(ReadlineError::Interrupted) => {
                if empty_ctrl_c.swap(false, Ordering::SeqCst) && !toggled.load(Ordering::SeqCst) {
                    if armed_exit.is_some_and(|t| t.elapsed().as_secs_f64() < 2.0) {
                        break;
                    }
                    armed_exit = Some(std::time::Instant::now());
                    println!("  (press Ctrl-C again to exit)");
                    continue;
                }
                if toggled.swap(false, Ordering::SeqCst) {
                    println!("  auto-approve {}", if auto.load(Ordering::SeqCst) { "ON (Shift+Tab again to turn off)" } else { "OFF" });
                }
                continue;
            }
            Err(ReadlineError::Eof) => break,
            Err(e) => return Err(e.into()),
        };
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }
        let _ = rl.add_history_entry(&line);
        let _ = rl.save_history(&hist);
        match line.as_str() {
            "/quit" | "/exit" | "/q" => break,
            "/help" | "/?" => println!("{HELP}"),
            "/list" => {
                for e in rt.store.experiments()? {
                    println!("  {} {:<24} {:>5} events", if e.name == exp { "*" } else { " " }, e.name, e.events);
                }
            }
            "/auto" => {
                let now = !auto.fetch_xor(true, Ordering::SeqCst);
                println!("auto-approve {}", if now { "ON: planner actions run without asking (still validated, still within budget, all recorded)" } else { "OFF: you approve according to the autonomy level" });
            }
            l if l.starts_with('/') => {
                run_slash(rt, &exp, l, learn_mode);
                // /new or /use selected another experiment: continue the chat there.
                if let Some(cur) = rt.store.setting("current")?.filter(|c| *c != exp) {
                    rt.release(&exp);
                    rt.open_for_write(&cur)?;
                    exp = cur;
                    let st = rt.state(&exp)?;
                    println!("now in experiment {exp} — autonomy {} — objective: {}", st.autonomy, st.objective);
                }
            }
            l => {
                let mut a = auto.load(Ordering::SeqCst);
                if let Err(e) = agent_turn(rt, &exp, Some(l.to_string()), learn_mode, &mut a) {
                    println!("error: {e:#}");
                }
                auto.store(a, Ordering::SeqCst); // [a] at an approval prompt can turn it on
            }
        }
    }
    println!("bye — everything is in the event log (`scientist log`, `scientist notebook`)");
    Ok(())
}

fn tokenize(line: &str) -> Vec<String> {
    let mut out = vec![];
    let mut cur = String::new();
    let mut quote: Option<char> = None;
    for ch in line.chars() {
        match (quote, ch) {
            (Some(q), c) if c == q => quote = None,
            (Some(_), c) => cur.push(c),
            (None, '"' | '\'') => quote = Some(ch),
            (None, c) if c.is_whitespace() => {
                if !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
            }
            (None, c) => cur.push(c),
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn run_slash(rt: &mut Runtime, exp: &str, line: &str, learn_mode: bool) {
    let mut args = vec!["scientist".to_string(), "--exp".to_string(), exp.to_string()];
    args.extend(tokenize(&line[1..]));
    match Cli::try_parse_from(&args) {
        Err(e) => println!("{e}"),
        Ok(mut c) => {
            match c.cmd {
                None => {
                    println!("unknown command {}; /help lists them", line.split_whitespace().next().unwrap_or(line));
                    return;
                }
                Some(cli::Cmd::Chat) => {
                    println!("already in a session");
                    return;
                }
                _ => {}
            }
            c.learn |= learn_mode;
            if let Err(e) = cli::dispatch(rt, c) {
                println!("error: {e:#}");
            }
        }
    }
}

fn show(rt: &Runtime, out: &PlannerOutput) {
    let x = &out.explanation;
    match &out.action {
        Action::Respond { message } => println!("\nplanner: {message}"),
        Action::AskHuman { question } => println!("\nplanner asks: {question}"),
        a @ (Action::SearchLiterature { .. } | Action::LookupKnown { .. }) => {
            println!("  researching: {}", a.describe());
            return;
        }
        a => {
            println!("\nPROPOSED   {}", a.describe());
            if !x.why.is_empty() {
                println!("WHY        {}", x.why);
            }
            for k in &x.what_we_know {
                println!("KNOWN      - {k}");
            }
            for k in &x.what_we_dont_know {
                println!("UNKNOWN    - {k}");
            }
            if !x.expected_observation.is_empty() {
                println!("EXPECT     {}", x.expected_observation);
            }
            if !x.would_falsify.is_empty() {
                println!("FALSIFIED  {}", x.would_falsify);
            }
        }
    }
    if !x.new_concepts.is_empty() {
        let tagged: Vec<String> = x.new_concepts.iter().map(|c| if learn::has_concept(&rt.cfg, c) { format!("{c} (/learn {})", learn::slug(c)) } else { c.clone() }).collect();
        println!("CONCEPTS   {}", tagged.join(", "));
    }
}

enum Approval {
    Yes,
    YesAll,
    No(String),
    Redirect(String),
    Command(String),
}

fn ask_approval() -> Result<Approval> {
    let mut ed = DefaultEditor::new()?;
    loop {
        let s = match ed.readline("  run it? [y] yes  [a] yes + auto-approve from now on  [n] no  /command = do this instead  other text = tell the planner > ") {
            Ok(s) => s,
            Err(ReadlineError::Interrupted) => return Ok(Approval::No("interrupted with Ctrl-C".into())),
            Err(_) => return Ok(Approval::No("input closed".into())),
        };
        let s = s.trim();
        match s {
            "" => continue,
            "y" | "yes" | "Y" => return Ok(Approval::Yes),
            "a" | "A" | "all" => return Ok(Approval::YesAll),
            "n" | "no" | "N" => {
                let r = ed.readline("  reason (optional, recorded) > ").unwrap_or_default();
                return Ok(Approval::No(if r.trim().is_empty() { "no reason given".into() } else { r.trim().to_string() }));
            }
            c if c.starts_with('/') => return Ok(Approval::Command(c.to_string())),
            t => return Ok(Approval::Redirect(t.to_string())),
        }
    }
}

pub fn agent_turn(rt: &mut Runtime, exp: &str, msg: Option<String>, learn_mode: bool, auto: &mut bool) -> Result<()> {
    let n = rt.cfg.planner.max_steps_per_turn;
    agent_turn_n(rt, exp, msg, learn_mode, n, auto)
}

/// `auto` = session auto-approve: every valid action runs without asking. Validation, budget
/// and stop rules are unchanged, and each approval is recorded as "auto-approve mode".
pub fn agent_turn_n(rt: &mut Runtime, exp: &str, msg: Option<String>, learn_mode: bool, max_steps: u32, auto: &mut bool) -> Result<()> {
    if let Some(m) = &msg {
        rt.append(exp, Actor::Human, vec![Event::UserMessage { text: m.clone() }])?;
    }
    let ui = Ui { learn: learn_mode };
    let mut feedback: Option<String> = None;
    let (mut retries, mut steps, mut lookups) = (0, 0, 0);
    // A question gets an answer before any tool run (enforced once per turn by the runtime,
    // because the planner model does not follow this rule reliably from the prompt alone).
    let asked = msg.as_deref().is_some_and(|m| m.trim_end().ends_with('?'));
    let mut nudged = false;
    while steps < max_steps {
        if rt.interrupted() {
            return interrupted(rt, exp, "the planner's turn");
        }
        let st = rt.state(exp)?;
        if st.autonomy == 0 {
            println!("  the planner is off at autonomy level 0 (`/level 1` turns it on); /commands still work");
            return Ok(());
        }
        if st.remaining().planner_calls == 0 {
            println!("  planner-call budget exhausted (`/budget --planner-calls N`)");
            return Ok(());
        }
        let events = rt.store.load_events(exp)?;
        eprint!("  planner thinking… ");
        let p = match planner::propose(&rt.cfg.planner, &st, &events, feedback.take(), rt.interrupt_flag()) {
            Ok(p) => p,
            Err(e) if e.is::<planner::Interrupted>() || rt.interrupted() => {
                eprintln!();
                return interrupted(rt, exp, "planner thinking");
            }
            Err(e) => {
                eprintln!();
                println!("  planner unavailable: {e:#}\n  you can keep working with /commands");
                return Ok(());
            }
        };
        let mut errors = p.errors.clone();
        if let Some(o) = &p.output {
            errors.extend(o.action.validate(&st, &rt.store));
        }
        eprintln!("{:.1} s ({} prompt + {} output tokens)", p.latency_ms as f64 / 1000.0, p.prompt_tokens, p.completion_tokens);
        rt.append(exp, Actor::Planner, vec![Event::PlannerProposed {
            output: p.raw.clone(),
            model: rt.cfg.planner.model.clone(),
            prompt_tokens: p.prompt_tokens,
            completion_tokens: p.completion_tokens,
            latency_ms: p.latency_ms,
            valid: errors.is_empty(),
            errors: errors.clone(),
        }])?;
        if !errors.is_empty() {
            retries += 1;
            println!("  runtime rejected the planner's proposal: {}", errors.join("; "));
            if retries > 2 {
                println!("  giving up after 3 invalid proposals; rephrase, or act directly with /commands");
                return Ok(());
            }
            feedback = Some(errors.join("; "));
            continue;
        }
        retries = 0;
        let out = p.output.unwrap();
        // Research lookups do not use up the turn's steps, but are capped so searching cannot loop.
        if matches!(out.action, Action::SearchLiterature { .. } | Action::LookupKnown { .. }) {
            lookups += 1;
            if lookups > 3 {
                feedback = Some("research limit for this request reached (3 lookups); answer or act with what you have".into());
                continue;
            }
        } else {
            if asked && !nudged && steps == 0 && !matches!(out.action, Action::Respond { .. } | Action::AskHuman { .. }) {
                nudged = true;
                rt.append(exp, Actor::Runtime, vec![Event::ActionRejected { action: serde_json::to_value(&out.action)?, reason: "runtime: Harsh asked a question; answer it before running tools".into() }])?;
                println!("  (runtime: you asked a question, so the planner is told to answer before running anything)");
                feedback = Some("Harsh asked a QUESTION. Answer it now with `respond` (OBSERVATION / LITERATURE / INTERPRETATION / NEXT), citing papers by id. Offer any build or computation as a NEXT step; do not run it.".into());
                continue;
            }
            steps += 1;
        }
        show(rt, &out);
        if matches!(out.action, Action::Respond { .. } | Action::AskHuman { .. }) {
            return Ok(());
        }
        if *auto && out.action.needs_approval(st.autonomy) {
            rt.append(exp, Actor::Human, vec![Event::ActionApproved { action: out.action.clone(), how: "auto-approve mode".into() }])?;
            println!("  (auto-approved)");
        } else if out.action.needs_approval(st.autonomy) {
            match ask_approval()? {
                Approval::Yes => rt.append(exp, Actor::Human, vec![Event::ActionApproved { action: out.action.clone(), how: "approved by human".into() }])?,
                Approval::YesAll => {
                    *auto = true;
                    println!("  auto-approve ON for this session (/auto turns it off)");
                    rt.append(exp, Actor::Human, vec![Event::ActionApproved { action: out.action.clone(), how: "auto-approve mode".into() }])?;
                }
                Approval::No(reason) => {
                    rt.append(exp, Actor::Human, vec![Event::ActionRejected { action: serde_json::to_value(&out.action)?, reason }])?;
                    println!("  not run. Say what you'd prefer, or use /commands.");
                    return Ok(());
                }
                Approval::Redirect(text) => {
                    rt.append(exp, Actor::Human, vec![Event::ActionRejected { action: serde_json::to_value(&out.action)?, reason: text.clone() }])?;
                    feedback = Some(format!("Harsh did not run that and said: {text}"));
                    continue;
                }
                Approval::Command(c) => {
                    rt.append(exp, Actor::Human, vec![Event::ActionRejected { action: serde_json::to_value(&out.action)?, reason: format!("ran {c} instead") }])?;
                    run_slash(rt, exp, &c, learn_mode);
                    return Ok(());
                }
            }
        } else {
            let lookup = matches!(out.action, Action::SearchLiterature { .. } | Action::LookupKnown { .. });
            let how = if lookup { "research lookup (no approval needed)".to_string() } else { format!("autonomy level {}", st.autonomy) };
            rt.append(exp, Actor::Planner, vec![Event::ActionApproved { action: out.action.clone(), how }])?;
            if !lookup {
                println!("  (runs without asking at autonomy level {})", st.autonomy);
            }
        }
        match actions::execute(rt, exp, &out.action, Actor::Planner, &ui) {
            Ok(done) if matches!(out.action, Action::SearchLiterature { .. }) => {
                let n = done.summary.lines().filter(|l| l.trim_start().starts_with('p')).count();
                println!("  found {n} paper(s); the planner reads them next (scientist inspect pN to read one)");
            }
            Ok(done) => {
                println!("RESULT     {}", done.summary);
                if let Some(t) = &done.text {
                    print!("{t}");
                }
                if learn_mode {
                    learn::render(rt, exp, &done)?;
                }
            }
            Err(e) => {
                println!("FAILED     {e:#}");
                feedback = Some(format!("the action failed: {e}"));
            }
        }
        if rt.interrupted() {
            return interrupted(rt, exp, &out.action.describe());
        }
        if matches!(out.action, Action::Terminate { .. }) {
            return Ok(());
        }
    }
    println!("  (planner used {max_steps} steps on this request; say 'continue' or give new direction)");
    Ok(())
}

/// Ctrl-C while the agent worked: record it (the planner sees it next turn), hand control back.
fn interrupted(rt: &mut Runtime, exp: &str, during: &str) -> Result<()> {
    rt.append(exp, Actor::Human, vec![Event::Interrupted { during: during.to_string() }])?;
    println!("  ⎿ Interrupted · tell the planner what to do instead");
    Ok(())
}
