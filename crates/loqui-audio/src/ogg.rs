//! A minimal Ogg muxer (RFC 3533): one logical stream, packets in order.
//! Enough to carry Opus (RFC 7845); nothing here reads Ogg.

/// Packets per audio page: one second of 20 ms Opus packets. Pages are the
/// unit a player seeks by; a second is what opusenc writes.
const PACKETS_PER_PAGE: usize = 50;

/// The most lacing values one page can hold.
const MAX_SEGMENTS: usize = 255;

const CONTINUED: u8 = 0x01;
const FIRST: u8 = 0x02;
const LAST: u8 = 0x04;

pub(crate) struct Writer {
    out: Vec<u8>,
    serial: u32,
    sequence: u32,
    /// The page being filled: its lacing values, body, and packet count.
    lacing: Vec<u8>,
    body: Vec<u8>,
    packets: usize,
    /// Set when the page's first packet began on the previous page.
    continued: bool,
    /// Where the last packet added ends, in the codec's granule units.
    granule: u64,
}

impl Writer {
    pub(crate) fn new(serial: u32) -> Self {
        Self { out: Vec::new(), serial, sequence: 0, lacing: Vec::new(), body: Vec::new(), packets: 0, continued: false, granule: 0 }
    }

    /// Adds a packet that ends at `granule`. `flush` closes the page after
    /// it, as Opus's two header packets each require.
    pub(crate) fn packet(&mut self, data: &[u8], granule: u64, flush: bool) {
        // A packet is laced as 255-byte segments, then one shorter segment
        // (possibly empty) that marks its end.
        let mut rest = data;
        loop {
            if self.lacing.len() == MAX_SEGMENTS {
                // Mid-packet: the page carries no packet end of its own
                // unless one finished on it, and the next page continues.
                let granule = if self.packets == 0 { u64::MAX } else { self.granule };
                self.page(granule, 0);
                self.continued = true;
            }
            let take = rest.len().min(255);
            self.lacing.push(take as u8);
            self.body.extend_from_slice(&rest[..take]);
            rest = &rest[take..];
            if take < 255 {
                break;
            }
        }
        self.packets += 1;
        self.granule = granule;
        if flush || self.packets == PACKETS_PER_PAGE {
            self.page(granule, 0);
        }
    }

    /// Closes the stream, marking its final page, and returns the bytes.
    pub(crate) fn finish(mut self) -> Vec<u8> {
        self.page(self.granule, LAST);
        self.out
    }

    fn page(&mut self, granule: u64, flags: u8) {
        let mut flags = flags;
        if self.sequence == 0 {
            flags |= FIRST;
        }
        if self.continued {
            flags |= CONTINUED;
        }
        let start = self.out.len();
        self.out.extend(b"OggS");
        self.out.push(0); // version
        self.out.push(flags);
        self.out.extend(granule.to_le_bytes());
        self.out.extend(self.serial.to_le_bytes());
        self.out.extend(self.sequence.to_le_bytes());
        self.out.extend([0; 4]); // CRC, filled below
        self.out.push(self.lacing.len() as u8);
        self.out.append(&mut self.lacing);
        self.out.append(&mut self.body);
        let crc = crc32(&self.out[start..]);
        self.out[start + 22..start + 26].copy_from_slice(&crc.to_le_bytes());
        self.sequence += 1;
        self.packets = 0;
        self.continued = false;
    }
}

/// Ogg's CRC-32: polynomial 0x04c11db7, MSB first, no reflection, zero
/// initial value and no final XOR.
fn crc32(bytes: &[u8]) -> u32 {
    const TABLE: [u32; 256] = {
        let mut table = [0u32; 256];
        let mut i = 0;
        while i < 256 {
            let mut crc = (i as u32) << 24;
            let mut bit = 0;
            while bit < 8 {
                crc = if crc & 0x8000_0000 != 0 { (crc << 1) ^ 0x04c1_1db7 } else { crc << 1 };
                bit += 1;
            }
            table[i] = crc;
            i += 1;
        }
        table
    };
    bytes.iter().fold(0, |crc, &b| (crc << 8) ^ TABLE[((crc >> 24) as u8 ^ b) as usize])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc_matches_the_ogg_check_value() {
        // CRC-32/MPEG-2 without its init and final XOR; the standard check
        // value for "123456789" under Ogg's parameters.
        assert_eq!(crc32(b"123456789"), 0x89a1_897f);
    }

    #[test]
    fn a_long_packet_spans_pages_and_marks_the_continuation() {
        let mut ogg = Writer::new(7);
        ogg.packet(&vec![1u8; 255 * 300], 960, false);
        let bytes = ogg.finish();
        let pages: Vec<usize> = bytes.windows(4).enumerate().filter(|(_, w)| w == b"OggS").map(|(i, _)| i).collect();
        assert_eq!(pages.len(), 2);
        // The first page ends mid-packet: no granule. The second continues it.
        assert_eq!(u64::from_le_bytes(bytes[6..14].try_into().unwrap()), u64::MAX);
        assert_eq!(bytes[pages[1] + 5], CONTINUED | LAST);
    }
}
