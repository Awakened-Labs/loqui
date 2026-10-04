//! Blends as people write them: whole percents, the accent first.
//!
//! The search moves weight in whole percents, so the blend it reports is the
//! blend it measured, not a rounding of one: `am_michael(65)+am_onyx(35)`,
//! readable, editable ("make it 70/30"), and inside the `\d+` weight grammar
//! [`crate::Blend`] parses.

/// Spread non-negative `weights` over 100 in steps of `step`, by largest
/// remainder, so the result sums to exactly 100. Ties go to the earlier
/// weight, which keeps the rounding deterministic. All-zero weights share
/// equally.
pub(crate) fn lattice(weights: &[f32], step: u32) -> Vec<u32> {
    let units = 100 / step;
    let total: f32 = weights.iter().sum();
    let raw: Vec<f32> =
        weights.iter().map(|w| if total > 0.0 { w / total * units as f32 } else { units as f32 / weights.len() as f32 }).collect();
    let mut out: Vec<u32> = raw.iter().map(|r| r.floor() as u32).collect();
    let mut short = units.saturating_sub(out.iter().sum());
    let mut order: Vec<usize> = (0..raw.len()).collect();
    order.sort_by(|&a, &b| (raw[b] - raw[b].floor()).total_cmp(&(raw[a] - raw[a].floor())).then(a.cmp(&b)));
    for i in order.into_iter().cycle() {
        if short == 0 {
            break;
        }
        out[i] += 1;
        short -= 1;
    }
    out.into_iter().map(|u| u * step).collect()
}

/// Which accent a blend speaks with. Kokoro takes it from the first voice
/// named, even at weight 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Lead<'a> {
    /// The heaviest voice leads.
    Heaviest,
    /// This voice leads, at weight 0 unless it is also in the blend: how a
    /// blend of American voices speaks with a British accent.
    Voice(&'a str),
}

/// `voices[i]` at `percents[i]`, as a spec: the lead first, then the rest by
/// weight, heaviest first (ties by name), voices at 0 left out unless they
/// lead. One voice at 100 with no other lead is written bare.
pub(crate) fn format(voices: &[&str], percents: &[u32], lead: Lead) -> String {
    let mut parts: Vec<(&str, u32)> = voices.iter().copied().zip(percents.iter().copied()).filter(|(_, p)| *p > 0).collect();
    parts.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
    if let Lead::Voice(lead) = lead {
        match parts.iter().position(|(v, _)| *v == lead) {
            Some(i) => {
                let part = parts.remove(i);
                parts.insert(0, part);
            }
            None => parts.insert(0, (lead, 0)),
        }
    }
    if let [(voice, 100)] = parts.as_slice() {
        return (*voice).to_owned();
    }
    parts.iter().map(|(v, p)| format!("{v}({p})")).collect::<Vec<_>>().join("+")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Blend;

    #[test]
    fn the_lattice_sums_to_100_and_keeps_the_proportions() {
        assert_eq!(lattice(&[0.7, 0.3], 5), [70, 30]);
        assert_eq!(lattice(&[1.0, 1.0, 1.0], 5), [35, 35, 30]);
        assert_eq!(lattice(&[0.0, 0.0], 5), [50, 50]);
        assert_eq!(lattice(&[0.333, 0.333, 0.334], 1), [33, 33, 34]);
        for w in [[0.12f32, 0.5, 0.38, 0.0], [0.01, 0.01, 0.01, 0.97], [0.26, 0.26, 0.24, 0.24]] {
            let p = lattice(&w, 5);
            assert_eq!(p.iter().sum::<u32>(), 100, "{w:?} -> {p:?}");
            assert!(p.iter().all(|x| x % 5 == 0), "{p:?}");
        }
    }

    #[test]
    fn the_heaviest_voice_leads_and_zero_weights_are_dropped() {
        let voices = ["af_sky", "af_bella", "af_heart"];
        assert_eq!(format(&voices, &[35, 65, 0], Lead::Heaviest), "af_bella(65)+af_sky(35)");
        assert_eq!(format(&voices, &[0, 100, 0], Lead::Heaviest), "af_bella");
        assert_eq!(format(&voices, &[50, 50, 0], Lead::Heaviest), "af_bella(50)+af_sky(50)", "ties by name");
    }

    #[test]
    fn another_accent_leads_at_weight_zero() {
        let voices = ["af_sky", "af_bella"];
        assert_eq!(format(&voices, &[100, 0], Lead::Voice("bf_emma")), "bf_emma(0)+af_sky(100)");
        assert_eq!(format(&voices, &[30, 70], Lead::Voice("af_sky")), "af_sky(30)+af_bella(70)", "a lead in the blend moves first");
    }

    /// Every spec this writes is one the engine reads back to the same mix.
    #[test]
    fn a_formatted_spec_parses_back_to_its_weights() {
        let voices = ["am_michael", "am_onyx", "bm_george"];
        for (percents, lead) in [([65, 35, 0], Lead::Heaviest), ([20, 30, 50], Lead::Voice("bf_emma")), ([0, 0, 100], Lead::Heaviest)] {
            let spec = format(&voices, &percents, lead);
            let blend: Blend = spec.parse().unwrap_or_else(|e| panic!("{spec}: {e}"));
            for (v, p) in voices.iter().zip(percents) {
                let parsed = blend.parts().iter().find(|(id, _)| id == v).map_or(0.0, |(_, w)| *w);
                assert!((parsed - p as f32 / 100.0).abs() < 1e-6, "{spec}: {v}");
            }
            if let Lead::Voice(lead) = lead {
                assert_eq!(blend.lead(), lead, "{spec}");
            }
        }
    }
}
