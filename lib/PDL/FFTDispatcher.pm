package PDL::FFTDispatcher;

# PDL::FFTDispatcher — one calling convention for FFTs in PDL, computed by
# PDL::PocketFFT (default), PDL::FFTW3 (when installed and chosen, or for float
# data) or PDL::FFT (only for the lengths its fftn.c transforms correctly).

use strict;
use warnings;
use Carp qw(croak);
use PDL;
use PDL::PocketFFT ();
use Exporter 'import';

our $VERSION = '0.001';

our @EXPORT_OK = qw(fft_c2c ifft_c2c fft_r2c ifft_c2r fft_inplace ifft_inplace
                    fft_backend fft_ok fft_len);
our %EXPORT_TAGS = (all => \@EXPORT_OK);

# 'auto'      : PDL::PocketFFT; PDL::FFTW3 for float / cfloat input when it is
#               installed (so the result stays in single precision)
# 'pocketfft' : PDL::PocketFFT for every call
# 'fftw3'     : PDL::FFTW3 for every call (croaks when it is not installed)
# 'pdl'       : PDL::FFT; croaks for a length it does not transform correctly
# The initial value comes from the environment variable PDL_FFTDISPATCHER_BACKEND
# when it is set (for example 'fftw3' on a machine whose FFTW is built with SIMD).
our $BACKEND = defined $ENV{PDL_FFTDISPATCHER_BACKEND} && length $ENV{PDL_FFTDISPATCHER_BACKEND}
             ? $ENV{PDL_FFTDISPATCHER_BACKEND} : 'auto';

my $NFACTOR = 11;    # size of factor[] in PDL's lib/PDL/FFT/fftn.c
my %NF;
my $HAVE_FFTW3;

sub _have_fftw3 {
    $HAVE_FFTW3 //= eval { require PDL::FFTW3; 1 } ? 1 : 0;
    return $HAVE_FFTW3;
}

# ---------------------------------------------------------------------------
# _nfactor($n): number of entries factorize() in PDL's fftn.c writes into
# factor[] for length $n. Same counts as the C code, without its O(n) loop:
#   - each factor 16 gives a 4, and each square p^2 of an odd prime gives a p;
#     these are written twice (the second half mirrors the first);
#   - if what is left is at most 4, it is one more entry (unless it is 1);
#   - otherwise a remaining factor 4 gives a 2 (also mirrored), and every
#     prime left (each appears once) is one entry.
# ---------------------------------------------------------------------------
sub _distinct_primes {
    my $n = shift;
    my $c = 0;
    for (my $p = 2; $p * $p <= $n; $p += ($p == 2 ? 1 : 2)) {
        next if $n % $p;
        $c++;
        $n /= $p while $n % $p == 0;
    }
    $c++ if $n > 1;
    return $c;
}

sub _nfactor {
    my $n  = shift;
    my $sq = 0;
    while ($n % 16 == 0) { $sq++; $n /= 16 }
    my $j = 3;
    do {
        while ($n % ($j * $j) == 0) { $sq++; $n /= $j * $j }
        $j += 2;
    } while ($j * $j <= $n);
    return 2 * $sq + ($n != 1 ? 1 : 0) if $n <= 4;
    my $two = 0;
    if ($n % 4 == 0) { $two = 1; $n /= 4 }
    return 2 * ($sq + $two) + _distinct_primes($n);
}

# fft_ok($n): true when PDL::FFT transforms length $n correctly
sub fft_ok {
    my $n = shift;
    croak "fft_ok: length must be a positive integer" unless defined $n && $n >= 1 && $n == int $n;
    return ($NF{$n} //= _nfactor($n)) <= $NFACTOR;
}

sub _is235 {
    my $r = shift;
    for my $p (2, 3, 5) { $r /= $p while $r % $p == 0 }
    return $r == 1;
}

# fft_len($n): smallest length >= $n with no prime factor other than 2, 3, 5
# that every backend (PDL::FFT included) transforms correctly
sub fft_len {
    my $n = shift;
    croak "fft_len: length must be a positive integer" unless defined $n && $n >= 1 && $n == int $n;
    for (my $m = $n; ; $m++) {
        return $m if _is235($m) && fft_ok($m);
    }
}

# fft_backend(\@lengths or $n, float => 0|1): the library that would transform
# these lengths under $BACKEND; croaks when none can
sub fft_backend {
    my ($len, %o) = @_;
    my @n = ref($len) eq 'ARRAY' ? @$len : ($len);
    my $b = $BACKEND;
    croak "PDL::FFTDispatcher: \$BACKEND must be 'auto', 'pocketfft', 'fftw3' or 'pdl', not '"
        . ($b // '') . "' (its initial value comes from PDL_FFTDISPATCHER_BACKEND when that is set)"
        unless defined $b && $b =~ /\A(?:auto|pocketfft|fftw3|pdl)\z/;
    if ($b eq 'auto') {
        return 'PDL::FFTW3' if $o{float} && _have_fftw3();
        return 'PDL::PocketFFT';
    }
    return 'PDL::PocketFFT' if $b eq 'pocketfft';
    if ($b eq 'fftw3') {
        croak "PDL::FFTDispatcher: \$BACKEND is 'fftw3' but PDL::FFTW3 is not installed" unless _have_fftw3();
        return 'PDL::FFTW3';
    }
    for my $m (@n) {
        croak sprintf "PDL::FFTDispatcher: PDL::FFT cannot transform length %d (it splits it into %d "
            . "factors, its limit is %d); set \$PDL::FFTDispatcher::BACKEND to 'auto'", $m, $NF{$m}, $NFACTOR
            unless fft_ok($m);
    }
    return 'PDL::FFT';
}

sub _check_rank {
    my ($x, $rank, $name) = @_;
    croak "$name: rank must be given" unless defined $rank;
    croak "$name: rank must be between 1 and " . $x->ndims . ", not $rank"
        unless $rank =~ /\A\d+\z/ && $rank >= 1 && $rank <= $x->ndims;
}

sub _is_float { my $t = shift->type; return $t == float() || $t == cfloat() }
sub _is_complex { my $t = shift->type; return $t == cdouble() || $t == cfloat() }

# PDL::FFT: complex transform of the first $rank dims of ($re, $im), in place
sub _pdlfft_reim {
    my ($re, $im, $rank, $inv) = @_;
    for my $d (0 .. $rank - 1) {
        next if $re->dim($d) == 1;
        my ($r, $i) = ($re->mv($d, 0), $im->mv($d, 0));
        $inv ? PDL::FFT::ifft($r, $i) : PDL::FFT::fft($r, $i);
    }
    return;
}

sub _pdlfft_c2c {
    my ($x, $rank, $inv) = @_;
    require PDL::FFT;
    my $z  = $x->convert(cdouble);
    my $re = $z->re->copy;
    my $im = $z->im->copy;
    _pdlfft_reim($re, $im, $rank, $inv);
    return czip($re, $im);
}

# ---------------------------------------------------------------------------
# complex <-> complex
# ---------------------------------------------------------------------------
sub _c2c {
    my ($x, $rank, $inv, $name) = @_;
    my $inplace = $x->is_inplace;
    $x->set_inplace(0) if $inplace;
    _check_rank($x, $rank, $name);
    my @n  = ($x->dims)[0 .. $rank - 1];
    my $be = fft_backend(\@n, float => _is_float($x));

    if ($be eq 'PDL::PocketFFT') {
        my $src = $inplace ? $x->inplace : $x;
        return $inv ? PDL::PocketFFT::icfftnd($src, $rank) : PDL::PocketFFT::cfftnd($src, $rank);
    }
    if ($be eq 'PDL::FFTW3') {
        if ($inplace && _is_complex($x)) {
            $inv ? PDL::FFTW3::ifftn($x, $rank, $x) : PDL::FFTW3::fftn($x, $rank, $x);
            return $x;
        }
        return $inv ? PDL::FFTW3::ifftn($x, $rank) : PDL::FFTW3::fftn($x, $rank);
    }
    my $y = _pdlfft_c2c($x, $rank, $inv);
    if ($inplace && _is_complex($x)) {
        $x .= $y;
        return $x;
    }
    return $y;
}

sub fft_c2c  { my ($x, $rank) = @_; return _c2c($x, $rank, 0, 'fft_c2c') }
sub ifft_c2c { my ($x, $rank) = @_; return _c2c($x, $rank, 1, 'ifft_c2c') }

# ---------------------------------------------------------------------------
# real -> complex (n0/2+1 frequencies along dim 0)
# ---------------------------------------------------------------------------
sub fft_r2c {
    my ($x, $rank) = @_;
    _check_rank($x, $rank, 'fft_r2c');
    croak "fft_r2c: input must be real" if _is_complex($x);
    my @n  = ($x->dims)[0 .. $rank - 1];
    my $be = fft_backend(\@n, float => _is_float($x));
    return PDL::PocketFFT::rfftnd($x, $rank) if $be eq 'PDL::PocketFFT';
    return PDL::FFTW3::rfftn($x, $rank)      if $be eq 'PDL::FFTW3';
    my $full = _pdlfft_c2c(czip($x->double, zeroes(double, $x->dims)), $rank, 0);
    return $full->slice('0:' . int($n[0] / 2))->copy;
}

# complex (n0/2+1 frequencies) -> real (n0); $n0 defaults to 2*(dim 0 - 1)
sub ifft_c2r {
    my ($X, $rank, $n0) = @_;
    _check_rank($X, $rank, 'ifft_c2r');
    my $m = $X->dim(0);
    $n0 = 2 * ($m - 1) unless defined $n0;
    croak "ifft_c2r: output length must be at least 1" unless $n0 >= 1;
    croak "ifft_c2r: input has $m frequencies, but output length $n0 needs " . (int($n0 / 2) + 1)
        unless $m == int($n0 / 2) + 1;
    my @n  = ($n0, ($X->dims)[1 .. $rank - 1]);
    my $be = fft_backend(\@n, float => _is_float($X));
    return PDL::PocketFFT::irfftnd($X, $rank, $n0) if $be eq 'PDL::PocketFFT';
    if ($be eq 'PDL::FFTW3') {
        my $type = _is_float($X) ? float() : double();
        my @out  = ($n0, ($X->dims)[1 .. $X->ndims - 1]);
        return PDL::FFTW3::irfftn($X, $rank, zeroes($type, @out));
    }
    # PDL::FFT: inverse along dims 1 .. rank-1 on the half spectrum, then
    # rebuild each row's full spectrum from Hermitian symmetry along dim 0
    require PDL::FFT;
    my $z  = $X->convert(cdouble);
    my $re = $z->re->copy;
    my $im = $z->im->copy;
    for my $d (1 .. $rank - 1) {
        next if $re->dim($d) == 1;
        PDL::FFT::ifft($re->mv($d, 0), $im->mv($d, 0));
    }
    # the imaginary parts of the zero-frequency term and (n0 even) the n0/2
    # term are ignored, as by PDL::PocketFFT and PDL::FFTW3
    $im->slice('(0)') .= 0;
    $im->slice('(' . ($n0 / 2) . ')') .= 0 if $n0 % 2 == 0;
    my $half = czip($re, $im);
    my $full = $half;
    if ($n0 > $m) {
        my $k   = sequence(long, $n0 - $m) + $m;     # k = m .. n0-1
        my $mir = $half->dice_axis(0, $n0 - $k)->conj;
        $full = $half->glue(0, $mir);
    }
    my ($fr, $fi) = ($full->re->copy, $full->im->copy);
    PDL::FFT::ifft($fr, $fi);
    return $fr;
}

# ---------------------------------------------------------------------------
# PDL::FFT::fft / ifft conventions: complex FFT along dim 0 of a real part and
# an imaginary part, in place
# ---------------------------------------------------------------------------
sub fft_inplace  { _reim_inplace(@_, 0) }
sub ifft_inplace { _reim_inplace(@_, 1) }

sub _reim_inplace {
    my ($re, $im, $inv) = @_;
    croak(($inv ? 'ifft' : 'fft') . "_inplace: re and im must have the same dims")
        unless join(',', $re->dims) eq join(',', $im->dims);
    my $be = fft_backend([$re->dim(0)], float => _is_float($re));
    if ($be eq 'PDL::FFT') {
        require PDL::FFT;
        $inv ? PDL::FFT::ifft($re, $im) : PDL::FFT::fft($re, $im);
        return;
    }
    my $z = czip($re, $im);
    $z = _c2c($z, 1, $inv, $inv ? 'ifft_inplace' : 'fft_inplace');
    $re .= $z->re;
    $im .= $z->im;
    return;
}

# method forms: $z->fft_c2c(2), $z->inplace->fft_c2c(2), $x->fft_r2c(1), ...
{
    no strict 'refs';
    *{"PDL::$_"} = \&{"PDL::FFTDispatcher::$_"} for qw(fft_c2c ifft_c2c fft_r2c ifft_c2r);
}

1;

__END__

=encoding utf8

=head1 NAME

PDL::FFTDispatcher - one calling convention for FFTs in PDL, over PDL::PocketFFT, PDL::FFTW3 and PDL::FFT

=head1 SYNOPSIS

    use PDL;
    use PDL::FFTDispatcher qw(fft_c2c ifft_c2c fft_r2c ifft_c2r);

    my $Z   = fft_c2c($z, 1);           # along dim 0, broadcasting over the rest
    my $IMG = fft_c2c($img, 2);         # 2-D, over dims 0 and 1
    $img->inplace->fft_c2c(2);          # in place (complex input)
    my $X   = fft_r2c($x, 1);           # real (n, ...) -> complex (n/2+1, ...)
    my $x2  = ifft_c2r($X, 1, $n);      # back to real (n, ...)

    $PDL::FFTDispatcher::BACKEND = 'fftw3';    # or 'pocketfft', 'pdl', 'auto' (default)

=head1 DESCRIPTION

The transforms run over the first C<$rank> dims and broadcast over the rest.
Forward transforms are unscaled, C<X[k] = sum_t x[t] exp(-2 pi i k t / n)>;
inverse transforms are scaled by 1/(n0*n1*...). The real-to-complex output
holds the n0/2+1 non-negative frequencies of dim 0.

The library is chosen by C<$PDL::FFTDispatcher::BACKEND>:

=over

=item auto (default)

PDL::PocketFFT (double precision). For float or cfloat input, PDL::FFTW3
when it is installed, so the result stays in single precision.

=item pocketfft

PDL::PocketFFT for every call.

=item fftw3

PDL::FFTW3 for every call; croaks when it is not installed.

=item pdl

PDL::FFT (PDL's own fftn.c) for every call, and a croak for any transformed
length it does not handle correctly (more than 11 factors in fftn.c, see
L</fft_ok>), instead of calling it.

=back

When the environment variable C<PDL_FFTDISPATCHER_BACKEND> is set (to one of
the names above) when the module is loaded, it gives the initial value of
C<$BACKEND>, so a machine can have its own default without changes to the
code, e.g. C<PDL_FFTDISPATCHER_BACKEND=fftw3> where FFTW is built with SIMD.
Assigning C<$BACKEND> in the code overrides it.

=head1 FUNCTIONS

=head2 fft_c2c($z, $rank), ifft_c2c($Z, $rank)

Complex to complex. Real input is taken as complex with zero imaginary part.
With C<< $z->inplace >> and complex input, the transform is written into
C<$z> itself and C<$z> is returned.

=head2 fft_r2c($x, $rank), ifft_c2r($X, $rank, $n0)

Real to complex and back. C<$n0> (the length of dim 0 of the real array)
defaults to 2*(dim 0 of $X - 1); pass it when it is odd. The imaginary parts
of the zero-frequency term (and of the n0/2 term when n0 is even) are
ignored.

=head2 fft_inplace($re, $im), ifft_inplace($re, $im)

Complex FFT along dim 0 of a real part and an imaginary part, in place, with
the conventions of PDL::FFT::fft and PDL::FFT::ifft.

=head2 fft_backend(\@lengths, float => 0|1)

The library (C<'PDL::PocketFFT'>, C<'PDL::FFTW3'> or C<'PDL::FFT'>) that
transforms these lengths under the current C<$BACKEND>.

=head2 fft_ok($n)

True when PDL::FFT transforms length C<$n> correctly: its fftn.c writes the
factors of C<$n> into an array of 11 entries without a bounds check, and for
longer factor lists it aborts the process or returns the input unchanged.

=head2 fft_len($n)

The smallest length C<< >= $n >> with no prime factor other than 2, 3 and 5
that every backend, PDL::FFT included, transforms correctly.

=cut
