//! Where the server listens, and the rules for each choice.
//!
//! Exposure is explicit and the narrow options are the easy ones:
//!
//! | Mode | Reachable by | Token | Transport |
//! |---|---|---|---|
//! | `unix` (default) | processes of the allowed uids on this host | optional | local socket, 0600 |
//! | `loopback:PORT` | anything on this host, any user | required | plaintext is fine |
//! | `tcp:ADDR:PORT` | one interface's network | required | TLS, or `--insecure-plaintext-network` |
//! | `all:PORT` | every interface, `--i-understand-this-exposes-all-interfaces` | required | TLS, or `--insecure-plaintext-network` |
//!
//! Loopback still needs a token because other local users, and web pages
//! through DNS rebinding, can reach 127.0.0.1.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::path::PathBuf;
use std::str::FromStr;

use crate::Error;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Listen {
    /// A Unix domain socket. `None` means the default path
    /// (`$XDG_RUNTIME_DIR/loqui/loqui.sock`).
    Unix(Option<PathBuf>),
    /// 127.0.0.1 (and nothing else) on this port.
    Loopback(u16),
    /// One specific, non-wildcard address.
    Interface(SocketAddr),
    /// 0.0.0.0 on this port: every interface.
    AllInterfaces(u16),
}

impl Default for Listen {
    fn default() -> Self {
        Self::Unix(None)
    }
}

impl FromStr for Listen {
    type Err = Error;

    /// `unix`, `unix:/path/to.sock`, `loopback:8100`, `tcp:192.168.1.5:8100`,
    /// `tcp:[fe80::1]:8100`, `all:8100`.
    fn from_str(s: &str) -> Result<Self, Error> {
        let bad = |why: &str| Error::Config(format!("--listen {s:?}: {why}"));
        let port = |p: &str| p.parse::<u16>().map_err(|_| bad("not a port number"));
        match s.split_once(':') {
            None if s == "unix" => Ok(Self::Unix(None)),
            Some(("unix", path)) if !path.is_empty() => Ok(Self::Unix(Some(PathBuf::from(path)))),
            Some(("loopback", p)) => Ok(Self::Loopback(port(p)?)),
            Some(("all", p)) => Ok(Self::AllInterfaces(port(p)?)),
            Some(("tcp", addr)) => {
                let addr: SocketAddr = addr.parse().map_err(|_| bad("expected ADDRESS:PORT, e.g. tcp:192.168.1.5:8100"))?;
                if addr.ip().is_unspecified() {
                    return Err(bad("a wildcard address listens everywhere; say so with all:PORT"));
                }
                if addr.ip().is_loopback() {
                    return Ok(Self::Loopback(addr.port()));
                }
                Ok(Self::Interface(addr))
            }
            _ => Err(bad("expected unix[:PATH], loopback:PORT, tcp:ADDRESS:PORT or all:PORT")),
        }
    }
}

impl std::fmt::Display for Listen {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unix(None) => f.write_str("unix (default path)"),
            Self::Unix(Some(p)) => write!(f, "unix:{}", p.display()),
            Self::Loopback(p) => write!(f, "loopback:{p}"),
            Self::Interface(a) => write!(f, "tcp:{a}"),
            Self::AllInterfaces(p) => write!(f, "all:{p}"),
        }
    }
}

/// The deliberate choices a wide listener needs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Acknowledgements {
    /// `--i-understand-this-exposes-all-interfaces`.
    pub all_interfaces: bool,
    /// `--insecure-plaintext-network`: serve a non-loopback address without
    /// TLS (a trusted LAN, a VPN, a TLS-terminating proxy in front).
    pub plaintext_network: bool,
}

impl Listen {
    /// Whether anything off this host can connect.
    pub fn is_network(&self) -> bool {
        matches!(self, Self::Interface(_) | Self::AllInterfaces(_))
    }

    pub fn is_unix(&self) -> bool {
        matches!(self, Self::Unix(_))
    }

    /// Refuses configurations that expose more than was clearly asked for.
    pub fn check(&self, ack: Acknowledgements, tls: bool) -> Result<(), Error> {
        if matches!(self, Self::AllInterfaces(_)) && !ack.all_interfaces {
            return Err(Error::Config(
                "all:PORT exposes loqui on every network interface. If that is intended, add \
                 --i-understand-this-exposes-all-interfaces; otherwise listen on loopback:PORT \
                 or one interface with tcp:ADDRESS:PORT"
                    .into(),
            ));
        }
        if self.is_network() && !tls && !ack.plaintext_network {
            return Err(Error::Config(format!(
                "{self} is reachable from the network, where the token would cross it in \
                 plaintext. Configure TLS (--tls-cert/--tls-key), put a TLS proxy in front and \
                 add --insecure-plaintext-network, or listen on loopback"
            )));
        }
        Ok(())
    }

    /// The TCP address to bind, for the TCP modes.
    pub fn socket_addr(&self) -> Option<SocketAddr> {
        match self {
            Self::Unix(_) => None,
            Self::Loopback(p) => Some(SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), *p)),
            Self::Interface(a) => Some(*a),
            Self::AllInterfaces(p) => Some(SocketAddr::new(IpAddr::V6(Ipv6Addr::UNSPECIFIED), *p)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_every_mode() {
        assert_eq!("unix".parse::<Listen>().unwrap(), Listen::Unix(None));
        assert_eq!("unix:/run/l.sock".parse::<Listen>().unwrap(), Listen::Unix(Some("/run/l.sock".into())));
        assert_eq!("loopback:8100".parse::<Listen>().unwrap(), Listen::Loopback(8100));
        assert_eq!("all:8100".parse::<Listen>().unwrap(), Listen::AllInterfaces(8100));
        assert_eq!("tcp:192.168.1.5:8100".parse::<Listen>().unwrap(), Listen::Interface("192.168.1.5:8100".parse().unwrap()));
        assert_eq!("tcp:127.0.0.1:9".parse::<Listen>().unwrap(), Listen::Loopback(9));
    }

    #[test]
    fn wildcards_must_be_spelled_all() {
        assert!("tcp:0.0.0.0:8100".parse::<Listen>().is_err());
        assert!("tcp:[::]:8100".parse::<Listen>().is_err());
        assert!("0.0.0.0:8100".parse::<Listen>().is_err());
        assert!("loopback:http".parse::<Listen>().is_err());
    }

    #[test]
    fn wide_listeners_need_acknowledgement() {
        let none = Acknowledgements::default();
        let all = Listen::AllInterfaces(8100);
        assert!(all.check(none, true).is_err(), "all interfaces needs its flag even with TLS");
        let ack = Acknowledgements { all_interfaces: true, plaintext_network: false };
        assert!(all.check(ack, false).is_err(), "and TLS or the plaintext flag");
        assert!(all.check(ack, true).is_ok());

        let lan = Listen::Interface("192.168.1.5:8100".parse().unwrap());
        assert!(lan.check(none, false).is_err());
        assert!(lan.check(Acknowledgements { plaintext_network: true, ..none }, false).is_ok());

        assert!(Listen::Loopback(8100).check(none, false).is_ok());
        assert!(Listen::Unix(None).check(none, false).is_ok());
    }
}
