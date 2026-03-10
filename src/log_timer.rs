use tracing_subscriber::fmt::time::FormatTime;

pub struct Iso8601Timer;

impl FormatTime for Iso8601Timer {
    fn format_time(&self, w: &mut tracing_subscriber::fmt::format::Writer<'_>) -> std::fmt::Result {
        let now =
            time::OffsetDateTime::now_local().unwrap_or_else(|_| time::OffsetDateTime::now_utc());
        write!(
            w,
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}{:+03}:{:02}",
            now.year(),
            now.month() as u8,
            now.day(),
            now.hour(),
            now.minute(),
            now.second(),
            now.offset().whole_hours(),
            now.offset().minutes_past_hour().abs()
        )
    }
}
