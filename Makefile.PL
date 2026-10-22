# Makefile.PL for PDL::FFTDispatcher — pure Perl; needs PDL::PocketFFT,
# uses PDL::FFTW3 when it is installed.
use strict;
use warnings;
use ExtUtils::MakeMaker;

WriteMakefile(
    NAME             => 'PDL::FFTDispatcher',
    VERSION_FROM     => 'lib/PDL/FFTDispatcher.pm',
    ABSTRACT         => 'one calling convention for FFTs in PDL, over PDL::PocketFFT, PDL::FFTW3 and PDL::FFT',
    LICENSE          => 'perl',
    MIN_PERL_VERSION => '5.016',
    PREREQ_PM        => { 'PDL' => '2.088', 'PDL::PocketFFT' => '0.001' },
    TEST_REQUIRES    => { 'Test::More' => '0.88' },
    META_MERGE       => {
        'meta-spec' => { version => 2 },
        prereqs     => { runtime => { recommends => { 'PDL::FFTW3' => '0.20' } } },
    },
);
