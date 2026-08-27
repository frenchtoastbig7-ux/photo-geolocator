# Third-party notices

## Vendored in this repository

### Leaflet 1.9.4 — BSD-2-Clause

`src/geoloc/web/static/leaflet.js`, `leaflet.css`, and `images/*` are
redistributed verbatim from [Leaflet](https://leafletjs.com). They are
vendored rather than loaded from a CDN so the workbench works with no
outbound requests.

> Copyright (c) 2010-2023, Volodymyr Agafonkin
> Copyright (c) 2010-2011, CloudMade
> All rights reserved.
>
> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions are met:
>
> 1. Redistributions of source code must retain the above copyright notice,
>    this list of conditions and the following disclaimer.
>
> 2. Redistributions in binary form must reproduce the above copyright
>    notice, this list of conditions and the following disclaimer in the
>    documentation and/or other materials provided with the distribution.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
> AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
> IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
> ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
> LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
> CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
> SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
> INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
> CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
> ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
> POSSIBILITY OF SUCH DAMAGE.

## Downloaded at runtime, not redistributed

### StreetCLIP — `geolocal/StreetCLIP`

Fetched from the Hugging Face Hub on first use by `geoloc prefetch-models`.
Not included in this repository. Review its model card and licence terms
before relying on it in any operational context.

## Services optionally queried

Used only when offline mode is off, and only ever with derived text — never
with image data. Each has its own usage policy, which you are responsible for
observing (including rate limits and attribution):

- **Nominatim** — geocoding of place names you choose to look up.
- **Overpass API** — OSM feature queries. Public instances are rate-limited;
  heavy use should point at your own instance via `GEOLOC_OVERPASS`.
- **OpenStreetMap tile servers** — basemap tiles. Map data © OpenStreetMap
  contributors, available under the Open Database License (ODbL).

### GeoNames — CC BY 4.0

Populated-place data reaches this project through the `geonamescache`
package. Source data is © GeoNames, licensed CC BY 4.0.
