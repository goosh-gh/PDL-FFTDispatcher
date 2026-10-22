# t/01-dispatch.t — PDL::FFTDispatcher: every backend against ground truths
# that do not use any FFT library (direct DFT applied axis by axis, plane
# waves with known spectra), backend selection, the PDL::FFT length guard,
# in-place and method forms, float handling and argument checks.
use strict;
use warnings;
use Test::More;
use PDL;
use PDL::FFTDispatcher qw(:all);

my $PI = 3.14159265358979323846;
my $HAVE_FFTW3 = eval { require PDL::FFTW3; 1 } ? 1 : 0;
my @BACKENDS = ('auto', 'pocket', 'pdl', ($HAVE_FFTW3 ? 'fftw3' : ()));
diag("PDL::FFTW3 not installed: the fftw3 backend is not tested") unless $HAVE_FFTW3;

sub maxerr { my ($p, $q) = @_; return ($p - $q)->abs->max->sclr }

# direct DFT along dim $d: X[k] = sum_t x[t] exp(sign 2 pi i (t k mod n) / n)
sub dft_axis {
    my ($x, $d, $sign) = @_;
    my $n  = $x->dim($d);
    my $tk = (xvals(long, $n, $n) * yvals(long, $n, $n)) % $n;      # (t, k)
    my $W  = exp(czip(zeroes($n, $n), $sign * 2 * $PI * $tk->double / $n));
    my $v  = $x->mv($d, 0);
    return ($v->dummy(1, $n) * $W)->sumover->mv(0, $d)->copy;
}

sub dft_nd {
    my ($x, $rank, $sign) = @_;
    my $y = czip($x->re->double, $x->im->double);
    $y = dft_axis($y, $_, $sign) for 0 .. $rank - 1;
    return $y;
}

# --- fft_ok: known lengths ---------------------------------------------
ok(fft_ok($_),  "fft_ok($_)")  for 1, 2, 1024, 1800000, 4194304, 300000;
ok(!fft_ok($_), "!fft_ok($_)") for 157464, 262440, 3600000, 8388608, 14400000;
is(fft_len(157464), 160000, 'fft_len(157464) = 160000 (2^8 5^4, 11 factors or fewer)');
ok(fft_ok(fft_len(14400000)), 'fft_len of an 8 h / 500 Hz record is a length PDL::FFT handles');

# --- every backend against the direct DFT ------------------------------
for my $be (@BACKENDS) {
    local $PDL::FFTDispatcher::BACKEND = $be;
    for my $dims ([12], [7, 5], [16, 9, 3], [6, 4, 5]) {
        my $label = "$be " . join('x', @$dims);
        my $z = czip(grandom(@$dims), grandom(@$dims));
        my $N = 1;
        $N *= $_ for @$dims;
        my $tol = 1e-12 * $N;
        for my $rank (1 .. @$dims) {
            my $M = 1;
            $M *= $dims->[$_] for 0 .. $rank - 1;
            cmp_ok(maxerr(fft_c2c($z, $rank), dft_nd($z, $rank, -1)), '<', $tol, "fft_c2c $label rank $rank");
            cmp_ok(maxerr(ifft_c2c($z, $rank), dft_nd($z, $rank, +1) / $M), '<', $tol, "ifft_c2c $label rank $rank");
            my $x = grandom(@$dims);
            my $Xr = dft_nd(czip($x, zeroes(@$dims)), $rank, -1)->slice('0:' . int($dims->[0] / 2));
            cmp_ok(maxerr(fft_r2c($x, $rank), $Xr), '<', $tol, "fft_r2c $label rank $rank");
            cmp_ok(maxerr(ifft_c2r(fft_r2c($x, $rank), $rank, $dims->[0]), $x), '<', 1e-13, "ifft_c2r(fft_r2c) $label rank $rank");
        }
    }
}

# --- plane wave on a 2-D grid, closed form, every backend ---------------
for my $be (@BACKENDS) {
    local $PDL::FFTDispatcher::BACKEND = $be;
    my ($n0, $n1, $k0, $k1) = (60, 48, 7, 11);
    my $ph = 2 * $PI * ((xvals(long, $n0, $n1) * $k0) % $n0)->double / $n0
           + 2 * $PI * ((yvals(long, $n0, $n1) * $k1) % $n1)->double / $n1;
    my $Z = fft_c2c(exp(czip(zeroes($n0, $n1), $ph)), 2);
    my $want = zeroes(cdouble, $n0, $n1);
    $want->set($k0, $k1, $n0 * $n1);
    cmp_ok(maxerr($Z, $want) / ($n0 * $n1), '<', 1e-13, "$be: 2-D plane wave -> N at its wavevector only");
}

# --- backend selection ---------------------------------------------------
{
    local $PDL::FFTDispatcher::BACKEND = 'auto';
    is(fft_backend([1024]), 'PDL::PocketFFT', 'auto: PocketFFT for double');
    is(fft_backend([157464]), 'PDL::PocketFFT', 'auto: PocketFFT for a length PDL::FFT cannot do');
    is(fft_backend([1024], float => 1), $HAVE_FFTW3 ? 'PDL::FFTW3' : 'PDL::PocketFFT',
        'auto: float input goes to FFTW3 when it is installed');
    is(fft_c2c(cfloat(1, 0, 0, 0), 1)->type, $HAVE_FFTW3 ? 'cfloat' : 'cdouble',
        'auto: cfloat input stays single precision with FFTW3, double otherwise');
    is(fft_c2c(pdl(1, 0, 0, 0), 1)->type, 'cdouble', 'auto: double input -> cdouble');
}
{
    local $PDL::FFTDispatcher::BACKEND = 'pdl';
    is(fft_backend([1024, 60]), 'PDL::FFT', 'pdl: PDL::FFT for lengths it handles');
    ok(!eval { fft_backend([1024, 157464]); 1 }, 'pdl: croaks when any transformed length overflows fftn.c');
    like($@, qr/cannot transform length 157464.*12 factors/, '... and says which length and why');
    # the guard: PDL::FFT must never be called on such a length (it would abort the process)
    ok(!eval { fft_c2c(zeroes(cdouble, 157464), 1); 1 }, 'pdl: fft_c2c of length 157464 croaks instead of running fftn.c');
    ok(!eval { fft_inplace(zeroes(262440), zeroes(262440)); 1 }, 'pdl: fft_inplace of length 262440 croaks');
    # a long but fine dim 1 is not checked when only dim 0 is transformed
    is(fft_c2c(zeroes(cdouble, 8, 1), 1)->dim(0), 8, 'pdl: only the transformed dims are checked');
}
{
    local $PDL::FFTDispatcher::BACKEND = 'nonsense';
    ok(!eval { fft_backend([8]); 1 }, 'unknown $BACKEND croaks');
}
SKIP: {
    skip 'PDL::FFTW3 is installed', 1 if $HAVE_FFTW3;
    local $PDL::FFTDispatcher::BACKEND = 'fftw3';
    ok(!eval { fft_backend([8]); 1 }, 'fftw3 backend croaks when PDL::FFTW3 is missing');
}

# --- in place, method forms, re/im API ------------------------------------
for my $be (@BACKENDS) {
    local $PDL::FFTDispatcher::BACKEND = $be;
    my $z   = czip(grandom(20, 12, 2), grandom(20, 12, 2));
    my $ref = dft_nd($z, 2, -1);
    my $w   = $z->copy;
    my $ret = $w->inplace->fft_c2c(2);
    cmp_ok(maxerr($w, $ref), '<', 1e-10, "$be: inplace fft_c2c writes into the array");
    ok(${$ret} == ${$w}, "$be: inplace fft_c2c returns the same array");
    ok(!$w->is_inplace, "$be: inplace flag cleared");
    my $x = grandom(10, 7);
    my $xc = $x->copy;
    cmp_ok(maxerr($x->fft_r2c(2)->ifft_c2r(2, 10), $x), '<', 1e-13, "$be: method forms fft_r2c / ifft_c2r");
    cmp_ok(maxerr($x, $xc), '==', 0, "$be: input of fft_r2c untouched");
    my ($re, $im) = (grandom(30), grandom(30));
    my ($r2, $i2) = ($re->copy, $im->copy);
    fft_inplace($r2, $i2);
    cmp_ok(maxerr(czip($r2, $i2), dft_nd(czip($re, $im), 1, -1)), '<', 1e-11, "$be: fft_inplace(re, im)");
    ifft_inplace($r2, $i2);
    cmp_ok(maxerr(czip($r2, $i2), czip($re, $im)), '<', 1e-13, "$be: ifft_inplace undoes it");
}

# --- argument checks -------------------------------------------------------
ok(!eval { fft_c2c(zeroes(cdouble, 4, 4)); 1 },    'fft_c2c croaks without rank');
ok(!eval { fft_c2c(zeroes(cdouble, 4, 4), 3); 1 }, 'fft_c2c croaks when rank > ndims');
ok(!eval { fft_r2c(zeroes(cdouble, 4), 1); 1 },    'fft_r2c croaks on complex input');
ok(!eval { ifft_c2r(zeroes(cdouble, 5), 1, 12); 1 }, 'ifft_c2r croaks when n0/2+1 != frequencies');
ok(!eval { fft_inplace(zeroes(4), zeroes(5)); 1 }, 'fft_inplace croaks on mismatched dims');

done_testing();
