Replaced the logo with the line-art mark and corrected how the assets are
derived from it. The background removal kept anti-aliased edges opaque in
their blended-with-white colour, which showed as a pale rim on any dark
surface and became a bright halo in the dark variant; it now solves for
coverage and colour separately. The dark variant also decided what counted
as greyscale using HLS saturation, which is unstable for near-black pixels
and left holes in the wordmark, and now uses absolute chroma.
