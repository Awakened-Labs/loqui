//! Trains the embedded POS tagger from spaCy-tagged text.
//!
//!     train_tagger TRAIN.jsonl [TRAIN.jsonl ...] --out data/tagger-en.bin [--eval EVAL.jsonl ...] [--iters 5] [--prune 0.001]
//!
//! Each input line is the JSON that `tools/parity/spacy_tag.py` writes: a
//! list of `[text, tag, whitespace]`. Training is an averaged perceptron
//! (Collins 2002; Honnibal's greedy tagger), deterministic for a given input
//! order. `--eval` files are only scored, never trained on.

use std::collections::HashMap;
use std::io::BufRead;

use loqui_g2p::tagger::{self, SPACE_TAG, Tagger};

type Sentence = Vec<(String, String)>;

fn read(path: &str) -> Result<Vec<Sentence>, Box<dyn std::error::Error>> {
    let file = std::io::BufReader::new(std::fs::File::open(path)?);
    let mut out = Vec::new();
    for line in file.lines() {
        let row: Vec<(String, String, String)> = serde_json::from_str(&line?)?;
        out.push(row.into_iter().map(|(text, tag, _)| (text, tag)).collect());
    }
    Ok(out)
}

#[derive(Default)]
struct Weight {
    value: f32,
    total: f64,
    stamp: u64,
}

struct Perceptron {
    classes: Vec<String>,
    class_ids: HashMap<String, u8>,
    feature_ids: HashMap<String, u32>,
    weights: Vec<HashMap<u8, Weight>>,
    step: u64,
}

impl Perceptron {
    fn feature(&mut self, name: String) -> u32 {
        let next = u32::try_from(self.feature_ids.len()).expect("features fit in u32");
        let id = *self.feature_ids.entry(name).or_insert(next);
        if id as usize == self.weights.len() {
            self.weights.push(HashMap::new());
        }
        id
    }

    fn predict(&self, features: &[u32]) -> u8 {
        let mut scores = vec![0f32; self.classes.len()];
        for &f in features {
            for (&class, w) in &self.weights[f as usize] {
                scores[class as usize] += w.value;
            }
        }
        let best = (0..scores.len()).max_by(|&a, &b| scores[a].total_cmp(&scores[b]).then(a.cmp(&b)));
        u8::try_from(best.unwrap_or(0)).expect("fewer than 256 tags")
    }

    fn update(&mut self, truth: u8, guess: u8, features: &[u32]) {
        self.step += 1;
        if truth == guess {
            return;
        }
        let step = self.step;
        for &f in features {
            for (class, delta) in [(truth, 1.0f32), (guess, -1.0)] {
                let w = self.weights[f as usize].entry(class).or_default();
                w.total += f64::from(w.value) * (step - w.stamp) as f64;
                w.stamp = step;
                w.value += delta;
            }
        }
    }

    fn averaged(&self, prune: f32) -> Vec<(String, Vec<(u8, f32)>)> {
        let mut names: Vec<(&String, &u32)> = self.feature_ids.iter().collect();
        names.sort();
        names
            .into_iter()
            .filter_map(|(name, &id)| {
                let mut row: Vec<(u8, f32)> = self.weights[id as usize]
                    .iter()
                    .map(|(&class, w)| {
                        let total = w.total + f64::from(w.value) * (self.step - w.stamp) as f64;
                        (class, (total / self.step as f64) as f32)
                    })
                    // Averaged weights this small rarely change an argmax
                    // and dominate the file size. `--prune` trades one for
                    // the other; the eval files show the cost.
                    .filter(|(_, w)| w.abs() >= prune)
                    .collect();
                row.sort_by_key(|&(class, _)| class);
                (!row.is_empty()).then(|| (name.clone(), row))
            })
            .collect()
    }
}

/// Words seen at least 20 times with one tag 97% of the time or more are
/// looked up instead of predicted, as in NLTK's tagger.
fn tagdict(sentences: &[Sentence], class_ids: &HashMap<String, u8>) -> Vec<(String, u8)> {
    let mut counts: HashMap<&str, HashMap<&str, u32>> = HashMap::new();
    for (word, tag) in sentences.iter().flatten() {
        *counts.entry(word).or_default().entry(tag).or_default() += 1;
    }
    let mut out: Vec<(String, u8)> = counts
        .into_iter()
        .filter_map(|(word, tags)| {
            let n: u32 = tags.values().sum();
            let (tag, &best) = tags.iter().max_by_key(|&(tag, count)| (*count, *tag))?;
            (n >= 20 && f64::from(best) / f64::from(n) >= 0.97 && word.len() < 256).then(|| (word.to_owned(), class_ids[*tag]))
        })
        .collect();
    out.sort();
    out
}

fn score(tagger: &Tagger, sentences: &[Sentence]) -> (usize, usize) {
    let (mut right, mut total) = (0, 0);
    for s in sentences {
        let words: Vec<&str> = s.iter().map(|(w, _)| w.as_str()).collect();
        for (guess, (_, truth)) in tagger.tag(&words).iter().zip(s) {
            right += usize::from(guess == truth);
            total += 1;
        }
    }
    (right, total)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mut train, mut eval, mut out, mut iters, mut prune) = (Vec::new(), Vec::new(), None, 5, 1e-3f32);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--out" => out = it.next().cloned(),
            "--eval" => eval.push(it.next().ok_or("--eval needs a file")?.clone()),
            "--iters" => iters = it.next().ok_or("--iters needs a number")?.parse()?,
            "--prune" => prune = it.next().ok_or("--prune needs a number")?.parse()?,
            path => train.push(path.to_owned()),
        }
    }
    let out = out.ok_or("--out is required")?;

    let mut sentences = Vec::new();
    for path in &train {
        sentences.extend(read(path)?);
    }
    let mut classes: Vec<String> = sentences.iter().flatten().map(|(_, t)| t.clone()).filter(|t| t != SPACE_TAG).collect();
    classes.sort();
    classes.dedup();
    let class_ids: HashMap<String, u8> =
        classes.iter().enumerate().map(|(i, c)| (c.clone(), u8::try_from(i).expect("fewer than 256 tags"))).collect();
    let tagdict = tagdict(&sentences, &class_ids);
    let lookup: HashMap<&str, u8> = tagdict.iter().map(|(w, c)| (w.as_str(), *c)).collect();
    eprintln!("{} sentences, {} tags, {} tagdict words", sentences.len(), classes.len(), tagdict.len());

    let mut model = Perceptron { classes: classes.clone(), class_ids, feature_ids: HashMap::new(), weights: Vec::new(), step: 0 };
    // A fixed-seed shuffle keeps training reproducible.
    let mut order: Vec<usize> = (0..sentences.len()).collect();
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    for iter in 0..iters {
        for i in (1..order.len()).rev() {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            order.swap(i, (seed >> 33) as usize % (i + 1));
        }
        let (mut right, mut total) = (0usize, 0usize);
        for &s in &order {
            let sentence = &sentences[s];
            let words: Vec<&str> = sentence.iter().map(|(w, _)| w.as_str()).collect();
            let context = tagger::context(&words);
            let (mut prev, mut prev2) = ("-START-".to_owned(), "-START2-".to_owned());
            for (i, (word, truth)) in sentence.iter().enumerate() {
                let guess = if truth == SPACE_TAG {
                    SPACE_TAG.to_owned()
                } else if let Some(&class) = lookup.get(word.as_str()) {
                    classes[class as usize].clone()
                } else {
                    let features: Vec<u32> = tagger::features(i, word, &context, &prev, &prev2)
                        .into_iter()
                        .map(|f| model.feature(f))
                        .collect();
                    let guess = model.predict(&features);
                    model.update(model.class_ids[truth], guess, &features);
                    classes[guess as usize].clone()
                };
                right += usize::from(&guess == truth);
                total += 1;
                prev2 = std::mem::replace(&mut prev, guess);
            }
        }
        eprintln!("iter {iter}: train accuracy {:.3}%", 100.0 * right as f64 / total as f64);
    }

    let bytes = tagger::encode(&classes, &tagdict, &model.averaged(prune));
    std::fs::write(&out, &bytes)?;
    eprintln!("wrote {out}: {} bytes", bytes.len());

    let tagger = Tagger::from_bytes(&bytes)?;
    for path in &eval {
        let (right, total) = score(&tagger, &read(path)?);
        eprintln!("{path}: {right}/{total} = {:.2}% agreement with spaCy", 100.0 * right as f64 / total as f64);
    }
    Ok(())
}
