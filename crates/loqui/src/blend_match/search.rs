//! The search: start from the mix of the nearest voices that best explains the
//! target, then move weight between pairs of them while the similarity of
//! what Kokoro actually says improves.
//!
//! Two stages, because the cheap one is only a guess. A blend's embedding is
//! not the mix of its voices' embeddings (on twelve pairs the two agreed at a
//! mean cosine of 0.80, and as low as 0.33 across genders), so the starting
//! point comes from the anchors and every step after it is measured by
//! speaking. The search runs on whole percents in steps of 20, then 10, then
//! 5, taking the first move that helps, so the result is a blend that was
//! itself spoken and measured, not an interpolation between two that were.

use std::collections::BTreeMap;
use std::ops::ControlFlow;

use super::spec::{self, Lead};
use crate::Error;

/// A move must gain this much to be taken: below it is the noise of a
/// slightly different waveform, not a better voice.
const MIN_GAIN: f32 = 0.001;

/// Euclidean projection onto the probability simplex
/// `{w : w_i >= 0, sum w_i = 1}`.
pub(crate) fn project(v: &[f32]) -> Vec<f32> {
    let mut u = v.to_vec();
    u.sort_by(|a, b| b.total_cmp(a));
    let mut sum = 0.0;
    let mut theta = 0.0;
    for (i, x) in u.iter().enumerate() {
        sum += x;
        let t = (sum - 1.0) / (i + 1) as f32;
        if x - t > 0.0 {
            theta = t;
        }
    }
    v.iter().map(|x| (x - theta).max(0.0)).collect()
}

/// Weights on the simplex whose mix of `anchors` is nearest `target`:
/// minimise `|sum w_i a_i - target|` by projected gradient descent, a fixed
/// number of steps from the uniform mix, so the same inputs always give the
/// same start.
pub(crate) fn least_squares(anchors: &[&[f32]], target: &[f32]) -> Vec<f32> {
    let k = anchors.len();
    // Lipschitz constant of the gradient, bounded by the Frobenius norm.
    let lipschitz: f32 = anchors.iter().map(|a| a.iter().map(|x| x * x).sum::<f32>()).sum::<f32>().max(f32::EPSILON);
    let mut w = vec![1.0 / k as f32; k];
    for _ in 0..200 {
        let residual: Vec<f32> = (0..target.len()).map(|d| anchors.iter().zip(&w).map(|(a, wi)| a[d] * wi).sum::<f32>() - target[d]).collect();
        let step: Vec<f32> = anchors
            .iter()
            .zip(&w)
            .map(|(a, wi)| wi - a.iter().zip(&residual).map(|(x, r)| x * r).sum::<f32>() / lipschitz)
            .collect();
        w = project(&step);
    }
    w
}

/// After each spoken candidate: how far the search is, and the best so far.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Evaluated {
    pub evaluations: usize,
    pub budget: usize,
    pub best: f32,
}

/// Speaks and scores candidates, at most `budget` of them, remembering each so
/// a revisit costs nothing. A `Break` from `progress` cancels the search.
pub(crate) struct Evaluator<'a> {
    objective: &'a mut dyn FnMut(&str) -> Result<f32, Error>,
    progress: &'a mut dyn FnMut(Evaluated) -> ControlFlow<()>,
    seen: BTreeMap<String, f32>,
    budget: usize,
    best: f32,
}

impl<'a> Evaluator<'a> {
    pub(crate) fn new(
        budget: usize,
        objective: &'a mut dyn FnMut(&str) -> Result<f32, Error>,
        progress: &'a mut dyn FnMut(Evaluated) -> ControlFlow<()>,
    ) -> Self {
        Self { objective, progress, seen: BTreeMap::new(), budget, best: f32::NEG_INFINITY }
    }

    /// The similarity of `spec`, or `None` once the budget is spent.
    pub(crate) fn score(&mut self, spec: &str) -> Result<Option<f32>, Error> {
        if let Some(&known) = self.seen.get(spec) {
            return Ok(Some(known));
        }
        if self.seen.len() >= self.budget {
            return Ok(None);
        }
        let similarity = (self.objective)(spec)?;
        self.seen.insert(spec.to_owned(), similarity);
        self.best = self.best.max(similarity);
        let report = Evaluated { evaluations: self.seen.len(), budget: self.budget, best: self.best };
        match (self.progress)(report) {
            ControlFlow::Continue(()) => Ok(Some(similarity)),
            ControlFlow::Break(()) => Err(Error::Cancelled),
        }
    }

    /// Allow `more` evaluations: a search that held some back for a last
    /// try spends them here.
    pub(crate) fn raise_budget(&mut self, more: usize) {
        self.budget += more;
    }

    pub(crate) fn has_seen(&self, spec: &str) -> bool {
        self.seen.contains_key(spec)
    }

    pub(crate) fn evaluations(&self) -> usize {
        self.seen.len()
    }
}

/// The best blend the search found: its percents (one per voice), its spec,
/// and its similarity as measured.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Found {
    pub percents: Vec<u32>,
    pub spec: String,
    pub similarity: f32,
}

/// Compass search on whole percents from `start`: for each pair of voices, in
/// rank order, move up to `step` points from one to the other and keep the
/// first move that helps by more than [`MIN_GAIN`]; when none does, halve the
/// step, down to 5. Stops early when the budget is spent.
pub(crate) fn refine<'v>(voices: &[&'v str], start: Vec<u32>, lead: impl Fn(&[u32]) -> Lead<'v>, eval: &mut Evaluator) -> Result<Found, Error> {
    let spec_of = |percents: &[u32]| spec::format(voices, percents, lead(percents));
    let first = spec_of(&start);
    let Some(similarity) = eval.score(&first)? else {
        return Ok(Found { percents: start, spec: first, similarity: f32::NEG_INFINITY });
    };
    let mut best = Found { percents: start, spec: first, similarity };
    for step in [20, 10, 5] {
        'improve: loop {
            for i in 0..voices.len() {
                for j in 0..voices.len() {
                    if i == j || best.percents[j] == 0 {
                        continue;
                    }
                    let mut candidate = best.percents.clone();
                    let moved = step.min(candidate[j]);
                    candidate[j] -= moved;
                    candidate[i] += moved;
                    let spec = spec_of(&candidate);
                    // A blend already spoken was not better when it was, or
                    // it would be the best now.
                    if eval.has_seen(&spec) {
                        continue;
                    }
                    match eval.score(&spec)? {
                        None => return Ok(best),
                        Some(similarity) if similarity > best.similarity + MIN_GAIN => {
                            best = Found { percents: candidate, spec, similarity };
                            continue 'improve;
                        }
                        Some(_) => {}
                    }
                }
            }
            break;
        }
    }
    Ok(best)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Blend;

    /// Deterministic unit vectors standing in for voice embeddings.
    fn anchors(k: usize, dim: usize) -> Vec<Vec<f32>> {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        (0..k)
            .map(|_| {
                let v: Vec<f32> = (0..dim)
                    .map(|_| {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        (state % 2001) as f32 / 1000.0 - 1.0
                    })
                    .collect();
                loqui_speaker::normalize(v)
            })
            .collect()
    }

    fn mix(anchors: &[Vec<f32>], weights: &[f32]) -> Vec<f32> {
        let dim = anchors[0].len();
        loqui_speaker::normalize((0..dim).map(|d| anchors.iter().zip(weights).map(|(a, w)| a[d] * w).sum()).collect())
    }

    const VOICES: [&str; 4] = ["v0", "v1", "v2", "v3"];

    /// A linear stand-in for speaking a spec and embedding it.
    fn linear_objective(anchors: &[Vec<f32>], target: Vec<f32>) -> impl FnMut(&str) -> Result<f32, Error> + '_ {
        move |spec: &str| {
            let blend: Blend = spec.parse().unwrap();
            let weights: Vec<f32> =
                VOICES.iter().map(|v| blend.parts().iter().find(|(id, _)| id == v).map_or(0.0, |(_, w)| *w)).collect();
            Ok(loqui_speaker::cosine(&mix(anchors, &weights), &target))
        }
    }

    #[test]
    fn projection_lands_on_the_simplex() {
        for v in [vec![0.2, 0.3, 0.5], vec![2.0, -1.0, 0.0], vec![-1.0, -1.0], vec![0.9, 0.9, 0.9]] {
            let p = project(&v);
            assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-5, "{v:?} -> {p:?}");
            assert!(p.iter().all(|x| *x >= 0.0));
        }
        assert_eq!(project(&[0.2, 0.3, 0.5]), [0.2, 0.3, 0.5], "already on it");
    }

    #[test]
    fn least_squares_recovers_a_linear_mix() {
        let anchors = anchors(4, 32);
        let truth = [0.6, 0.3, 0.1, 0.0];
        let target: Vec<f32> = (0..32).map(|d| anchors.iter().zip(&truth).map(|(a, w)| a[d] * w).sum()).collect();
        let refs: Vec<&[f32]> = anchors.iter().map(Vec::as_slice).collect();
        let w = least_squares(&refs, &target);
        for (got, want) in w.iter().zip(truth) {
            assert!((got - want).abs() < 0.05, "{w:?}");
        }
    }

    #[test]
    fn refine_recovers_the_weights_of_a_linear_blend_from_a_poor_start() {
        let anchors = anchors(4, 32);
        let target = mix(&anchors, &[0.7, 0.0, 0.3, 0.0]);
        let mut objective = linear_objective(&anchors, target);
        let mut progress = |_: Evaluated| ControlFlow::Continue(());
        let mut eval = Evaluator::new(60, &mut objective, &mut progress);
        let found = refine(&VOICES, vec![25, 25, 25, 25], |_| Lead::Heaviest, &mut eval).unwrap();
        assert!(found.similarity > 0.995, "{found:?}");
        assert!(found.percents[0].abs_diff(70) <= 10 && found.percents[2].abs_diff(30) <= 10, "{found:?}");
        assert!(eval.evaluations() <= 60);
    }

    #[test]
    fn refine_is_deterministic_and_keeps_to_its_budget() {
        let anchors = anchors(4, 32);
        let run = |budget: usize| {
            let mut objective = linear_objective(&anchors, mix(&anchors, &[0.2, 0.5, 0.0, 0.3]));
            let mut calls = 0;
            let mut progress = |e: Evaluated| {
                calls = e.evaluations;
                ControlFlow::Continue(())
            };
            let mut eval = Evaluator::new(budget, &mut objective, &mut progress);
            let found = refine(&VOICES, vec![100, 0, 0, 0], |_| Lead::Heaviest, &mut eval).unwrap();
            (found, eval.evaluations(), calls)
        };
        let (a, used, reported) = run(7);
        assert_eq!(used, 7, "the budget is spent, not exceeded");
        assert_eq!(reported, 7, "progress after every evaluation");
        assert_eq!(run(7).0, a, "the same inputs, the same blend");
    }

    #[test]
    fn a_break_from_progress_cancels_the_search() {
        let anchors = anchors(4, 32);
        let mut objective = linear_objective(&anchors, mix(&anchors, &[0.5, 0.5, 0.0, 0.0]));
        let mut progress = |e: Evaluated| if e.evaluations >= 3 { ControlFlow::Break(()) } else { ControlFlow::Continue(()) };
        let mut eval = Evaluator::new(36, &mut objective, &mut progress);
        let err = refine(&VOICES, vec![25, 25, 25, 25], |_| Lead::Heaviest, &mut eval).unwrap_err();
        assert!(matches!(err, Error::Cancelled), "{err:?}");
    }
}
