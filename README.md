# PDL::FFTDispatcher

One calling convention for FFTs in PDL. The transform is computed by
PDL::PocketFFT (default), PDL::FFTW3 (when installed and chosen, or for float
data), or PDL::FFT (only for the lengths it transforms correctly).

```perl
use PDL;
use PDL::FFTDispatcher qw(fft_c2c ifft_c2c fft_r2c ifft_c2r);

my $Z   = fft_c2c($z, 1);            # along dim 0, broadcasting over the rest
my $IMG = fft_c2c($img, 2);          # 2-D, over dims 0 and 1
$img->inplace->fft_c2c(2);           # in place (complex input)
my $X   = fft_r2c($x, 1);            # real (n, ...) -> complex (n/2+1, ...)
my $x2  = ifft_c2r($X, 1, $n);       # back to real (n, ...)

$PDL::FFTDispatcher::BACKEND = 'fftw3';   # 'auto' (default), 'pocketfft', 'fftw3', 'pdl'
```

- The transforms run over the first `$rank` dims and broadcast over the
  rest. Forward transforms are unscaled; inverse transforms are scaled by
  1/(n0*n1*...). The real-to-complex output holds the n0/2+1 non-negative
  frequencies of dim 0.
- `auto` uses PDL::PocketFFT, and PDL::FFTW3 for float / cfloat input when it
  is installed (the result stays in single precision).
- `pdl` uses PDL::FFT and croaks, instead of calling it, for any transformed
  length that PDL::FFT's fftn.c does not handle (more than 11 factors; fftn.c
  writes them into an 11-entry array without a bounds check and then aborts
  the process or returns the input unchanged).
- The environment variable `PDL_FFTDISPATCHER_BACKEND`, when set at load
  time, gives the initial value of `$BACKEND` (e.g. `fftw3` on a machine
  whose FFTW is built with SIMD); assigning `$BACKEND` in code overrides it.
- `fft_inplace($re, $im)` and `ifft_inplace($re, $im)` keep the calling
  convention of PDL::FFT::fft and ifft.
- `fft_ok($n)` tells whether PDL::FFT handles length `$n`; `fft_len($n)` is
  the smallest length `>= $n` with prime factors 2, 3 and 5 only that every
  backend handles.

## Build

```
perl Makefile.PL
make
make test
make install
```

Requires PDL 2.088 or later and PDL::PocketFFT. PDL::FFTW3 is used when it is
installed.

## Tests

`t/01-dispatch.t` checks every backend (auto, pocketfft, pdl, and fftw3 when
PDL::FFTW3 is installed) against a direct DFT applied axis by axis (1 to 3
dims, every rank) and a 2-D plane wave with a known spectrum; the factor
count against known lengths; backend selection, including float input; that
the pdl backend croaks rather than calling PDL::FFT on lengths 157464 and
262440; in-place, method and re/im forms; the argument checks; and the
initial `$BACKEND` from `PDL_FFTDISPATCHER_BACKEND` (in child perls). Without
PDL::FFTW3 the fftw3 cases are skipped and the rest pass.

## License

Same terms as Perl itself.
