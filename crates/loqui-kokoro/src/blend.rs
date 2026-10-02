//! Voice specs: one Kokoro voice, or several mixed into one.
//!
//! The syntax is open-speech's, kept exactly: `af_heart`, an OpenAI name
//! such as `alloy`, or a blend such as `af_bella(2)+af_sky(1)`. A blend is
//! the weighted mean of its voices' packs. That is linear in the weights, so
//! a named blend used inside another expands into its own parts exactly.

use std::str::FromStr;

use crate::{Error, resolve_alias};

/// A voice to speak in: one or more voice packs whose weights sum to one.
///
/// The first part sets the accent, as in open-speech; it may weigh nothing
/// (`bf_emma(0)+af_sky` is Sky's voice with a British accent). Each id
/// appears once, in the order it was first written.
#[derive(Debug, Clone, PartialEq)]
pub struct Blend {
    parts: Vec<(String, f32)>,
}

impl Blend {
    /// Parses a voice spec. Each component is resolved in turn: through
    /// `named` (an application's own voices, which may remap an OpenAI
    /// name), then as an OpenAI name, and otherwise taken as a pack id.
    ///
    /// Weights are plain decimals and are normalised to sum to one; if they
    /// are all zero, the components share equally. Pack ids are limited to
    /// ASCII letters, digits and `_`, because they become file names.
    pub fn parse<'a>(spec: &str, named: impl Fn(&str) -> Option<&'a Blend>) -> Result<Self, Error> {
        let components = spec.split('+').map(component).collect::<Result<Vec<_>, _>>()?;
        let total: f32 = components.iter().map(|(_, weight)| weight).sum();
        if !total.is_finite() {
            return Err(too_heavy());
        }
        let mut parts: Vec<(String, f32)> = Vec::with_capacity(components.len());
        let mut add = |id: &str, weight: f32| match parts.iter_mut().find(|(seen, _)| seen == id) {
            Some((_, sum)) => *sum += weight,
            None => parts.push((id.to_owned(), weight)),
        };
        for &(id, weight) in &components {
            let share = if total == 0.0 { 1.0 / components.len() as f32 } else { weight / total };
            match named(id) {
                Some(blend) => blend.parts.iter().for_each(|(id, weight)| add(id, weight * share)),
                None => add(resolve_alias(id), share),
            }
        }
        Ok(Self { parts })
    }

    /// Every part, weights summing to one, the accent-setting part first.
    pub fn parts(&self) -> &[(String, f32)] {
        &self.parts
    }

    /// The part whose accent the blend speaks with.
    pub fn lead(&self) -> &str {
        &self.parts[0].0
    }

    /// The parts that are heard: all but those weighing nothing, which can
    /// only set the accent.
    pub fn audible(&self) -> impl Iterator<Item = (&str, f32)> {
        self.parts.iter().filter(|(_, weight)| *weight > 0.0).map(|(id, weight)| (id.as_str(), *weight))
    }
}

/// Parses a spec with no named voices of its own: pack ids and OpenAI names.
impl FromStr for Blend {
    type Err = Error;

    fn from_str(spec: &str) -> Result<Self, Error> {
        Self::parse(spec, |_| None)
    }
}

/// One `id` or `id(weight)`, as open-speech's `([a-zA-Z0-9_]+)(?:\((\d+(?:\.\d+)?)\))?`.
fn component(part: &str) -> Result<(&str, f32), Error> {
    let part = part.trim();
    let (id, weight) = match part.strip_suffix(')').and_then(|p| p.split_once('(')) {
        Some((id, weight)) => (id, Some(weight)),
        None => (part, None),
    };
    // The id is not echoed: it is caller input that failed validation.
    if id.is_empty() || !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return Err(Error::Voice("voice ids may contain only ASCII letters, digits and _".into()));
    }
    let weight = match weight {
        None => 1.0,
        Some(w) if is_decimal(w) => w.parse::<f32>().ok().filter(|w| w.is_finite()).ok_or_else(too_heavy)?,
        Some(_) => return Err(Error::Voice("voice weights are plain decimals, as in af_bella(2)+af_sky(0.5)".into())),
    };
    Ok((id, weight))
}

fn is_decimal(s: &str) -> bool {
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    match s.split_once('.') {
        Some((whole, fraction)) => digits(whole) && digits(fraction),
        None => digits(s),
    }
}

fn too_heavy() -> Error {
    Error::Voice("voice weights are too large".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weights(spec: &str) -> Vec<(String, f32)> {
        spec.parse::<Blend>().unwrap().parts
    }

    fn close(actual: &[(String, f32)], expected: &[(&str, f32)]) -> bool {
        actual.len() == expected.len() && actual.iter().zip(expected).all(|((a, x), (b, y))| a == b && (x - y).abs() < 1e-6)
    }

    #[test]
    fn single_voices_and_openai_names() {
        assert!(close(&weights("af_heart"), &[("af_heart", 1.0)]));
        assert!(close(&weights("alloy"), &[("af_heart", 1.0)]));
        assert!(close(&weights("  af_sky "), &[("af_sky", 1.0)]));
        assert!(close(&weights("af_sky(3)"), &[("af_sky", 1.0)]));
    }

    #[test]
    fn weights_are_normalised_as_open_speech_does() {
        assert!(close(&weights("af_bella+af_sky"), &[("af_bella", 0.5), ("af_sky", 0.5)]));
        assert!(close(&weights("af_bella(2)+af_sky(1)"), &[("af_bella", 2.0 / 3.0), ("af_sky", 1.0 / 3.0)]));
        assert!(close(&weights("am_puck(1)+am_liam(1)+am_onyx(0.5)"), &[("am_puck", 0.4), ("am_liam", 0.4), ("am_onyx", 0.2)]));
        assert!(close(&weights("af_bella(1.5) + af_sky(0.5)"), &[("af_bella", 0.75), ("af_sky", 0.25)]));
    }

    #[test]
    fn all_zero_weights_share_equally_counting_each_component() {
        assert!(close(&weights("af_bella(0)+af_sky(0)"), &[("af_bella", 0.5), ("af_sky", 0.5)]));
        // Equal shares go to components as written, then duplicates merge.
        assert!(close(&weights("af_bella(0)+af_sky(0)+af_bella(0)"), &[("af_bella", 2.0 / 3.0), ("af_sky", 1.0 / 3.0)]));
    }

    #[test]
    fn duplicates_merge_in_first_appearance_order() {
        assert!(close(&weights("af_sky+af_sky"), &[("af_sky", 1.0)]));
        assert!(close(&weights("af_bella(1)+af_sky(1)+af_bella(2)"), &[("af_bella", 0.75), ("af_sky", 0.25)]));
        assert!(close(&weights("alloy+af_heart"), &[("af_heart", 1.0)]));
    }

    #[test]
    fn a_weightless_lead_sets_only_the_accent() {
        let blend: Blend = "bf_emma(0)+af_sky".parse().unwrap();
        assert_eq!(blend.lead(), "bf_emma");
        assert_eq!(blend.audible().collect::<Vec<_>>(), [("af_sky", 1.0)]);
    }

    #[test]
    fn openai_names_resolve_inside_blends() {
        assert!(close(&weights("alloy+af_sky"), &[("af_heart", 0.5), ("af_sky", 0.5)]));
        assert!(close(&weights("shimmer(3)+onyx(1)"), &[("af_bella", 0.75), ("am_michael", 0.25)]));
    }

    #[test]
    fn named_voices_expand_in_proportion() {
        let will: Blend = "am_puck(2)+am_liam(2)+am_onyx(1)".parse().unwrap();
        let named = |id: &str| (id == "will").then_some(&will);
        let blend = Blend::parse("will(2)+af_sky(1)", named).unwrap();
        assert!(close(&blend.parts, &[("am_puck", 4.0 / 15.0), ("am_liam", 4.0 / 15.0), ("am_onyx", 2.0 / 15.0), ("af_sky", 1.0 / 3.0)]));
        let blend = Blend::parse("will(0)+af_sky(0)", named).unwrap();
        assert!(close(&blend.parts, &[("am_puck", 0.2), ("am_liam", 0.2), ("am_onyx", 0.1), ("af_sky", 0.5)]));
    }

    #[test]
    fn a_name_takes_precedence_over_an_openai_name() {
        let nova: Blend = "af_nova(3)+af_sky(1)".parse().unwrap();
        let blend = Blend::parse("nova", |id| (id == "nova").then_some(&nova)).unwrap();
        assert!(close(&blend.parts, &[("af_nova", 0.75), ("af_sky", 0.25)]));
    }

    #[test]
    fn rejects_what_open_speech_rejects() {
        let bad =
            ["", "+", "af_heart+", "af_heart++af_sky", "af_heart(1e2)", "af_heart(.5)", "af_heart(1.)", "af_heart(+1)", "af_heart(-1)"];
        let bad2 = ["af_heart (2)", "af_heart( 2)", "af_heart(2", "af_heart(x)", "af_heart(2)(3)", "af-heart", "af_heart.bin", "a b"];
        for spec in bad.into_iter().chain(bad2) {
            assert!(spec.parse::<Blend>().is_err(), "{spec:?} should be rejected");
        }
    }

    #[test]
    fn ids_cannot_name_paths() {
        for spec in ["../etc/passwd", "af/heart", "af_heart+../../x", "voices\\af_heart"] {
            assert!(spec.parse::<Blend>().is_err(), "{spec:?} should be rejected");
        }
    }

    #[test]
    fn weights_too_large_to_sum_are_rejected_not_silenced() {
        // 3e38 parses (f32 tops out near 3.4e38), but two of them do not sum.
        let huge = format!("3{}", "0".repeat(38));
        assert!(format!("af_heart({huge})+af_sky({huge})").parse::<Blend>().is_err());
        assert!(format!("af_heart({huge})+af_heart({huge})").parse::<Blend>().is_err());
        assert!(format!("af_heart(1{})", "0".repeat(40)).parse::<Blend>().is_err());
        assert!(format!("af_heart({huge})+af_sky(1)").parse::<Blend>().is_ok());
    }

    #[test]
    fn errors_do_not_echo_the_spec() {
        for spec in ["zz/secret", "zz_secret(oops)"] {
            let message = spec.parse::<Blend>().unwrap_err().to_string();
            assert!(!message.contains("secret"), "{message}");
        }
    }
}
