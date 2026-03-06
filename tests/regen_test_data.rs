/// Regenerate test_data/Qwen3-4B.json with current engine outputs (greedy decoding).
use pegainfer::sampler::SamplingParams;
use pegainfer::server_engine::{CompleteRequest, RealServerEngine, ServerEngine};

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/Qwen3-4B");
const TEST_DATA_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/test_data/Qwen3-4B.json");

struct Case {
    name: &'static str,
    prompt: &'static str,
    max_new_tokens: usize,
}

const CASES: &[Case] = &[
    Case { name: "tell_story", prompt: "Tell me a story", max_new_tokens: 50 },
    Case { name: "my_name", prompt: "My name is", max_new_tokens: 50 },
    Case { name: "math", prompt: "What is 2 + 2?", max_new_tokens: 30 },
    Case { name: "chinese_weather", prompt: "今天天气真好", max_new_tokens: 50 },
    Case { name: "chinese_capital", prompt: "请介绍一下中国的首都", max_new_tokens: 50 },
    Case { name: "python_code", prompt: "Write a Python function to reverse a string", max_new_tokens: 50 },
];

#[test]
fn regen_test_data() {
    pegainfer::logging::init_stderr("info");
    let mut engine = RealServerEngine::load(MODEL_PATH, 42).expect("Failed to load model");

    let mut cases_json = Vec::new();
    for case in CASES {
        let req = CompleteRequest {
            prompt: case.prompt.to_string(),
            max_tokens: case.max_new_tokens,
            sampling: SamplingParams { temperature: 0.0, top_k: 0, top_p: 1.0 },
        };
        let resp = engine.complete(req).expect("complete failed");
        let output = resp.text;
        eprintln!("[{}] prompt={:?} output={:?}", case.name, case.prompt, output);
        cases_json.push(serde_json::json!({
            "name": case.name,
            "prompt": case.prompt,
            "max_new_tokens": case.max_new_tokens,
            "output": output,
        }));
    }

    let data = serde_json::json!({
        "model_name": "Qwen3-4B",
        "engine": "pegainfer",
        "cases": cases_json,
    });
    let json = serde_json::to_string_pretty(&data).unwrap();
    std::fs::write(TEST_DATA_PATH, json).expect("Failed to write test data");
    eprintln!("Wrote {TEST_DATA_PATH}");
}
