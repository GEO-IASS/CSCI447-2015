#!/usr/bin/perl

=for comment
Machine Learning Assignment 2 Log File Parser
Authors: Clint Cooper, Emily Rohrbough, Leah Thompson
Date: 10/10/15

Takes the generated .log files that contain the results of the tests 
on the NN and prints the data organized to a tsv file.
=cut

use strict;
use warnings;
#use diagnostics;

my $folder = "Results";

open(my $output, '>', "results.tsv") or die "Unable to write new data file";
my @list = process_files($folder);

foreach my $file (@list){
	if (-f $file) {
		open(INFILE, $file) or die "Cannot open $_!.\n";
		#$file =~ s/$folder\///;
		my $line;
		$line = <INFILE>;
		$line = <INFILE>;
		my $activationSet;
		$activationSet = $line;
		if ($activationSet) {chomp($activationSet)}
		$line = <INFILE>;
		my $activationOutput;
		$activationOutput = $line;
		if ($activationOutput) {$activationOutput =~ s/(OUTPUT ACTIVATION: )//}
		if ($activationOutput) {chomp($activationOutput)}
		$line = <INFILE>;
		$line = <INFILE>;
		$line = <INFILE>;
		while($line = <INFILE>) {
			my $testInputs = $line;
			chomp($testInputs);
			#print "In: $testInputs\n";
			$line = <INFILE>;
			my $testOutput = $line;
			chomp($testOutput);
			$testOutput =~ s/\[//;
			$testOutput =~ s/\]//;
			#print "Out: $testOutput\n";
			$line = <INFILE>;
			my $RBOutput = $line;
			chomp($RBOutput);
			$RBOutput =~ s/(RB OUTPUT: )//;
			#print "RB: $RBOutput\n";
			my $error = (abs(int($testOutput) - (int($RBOutput))) / (int($RBOutput) + 0.000001));
			chomp($error);
			printf $output "$activationSet\t$activationOutput\t$testInputs\t%.2f\t$RBOutput\t%.2f\n", $testOutput, $error;
		}
	}
}

sub process_files {    
	my $path = shift;
	opendir (DIR, $path) or die "Unable to open $path: $!";
	my @files =
		map { $path . '/' . $_ }
		grep { !/^\.{1,2}$/ } # Don't climb up
		readdir (DIR);

	closedir (DIR);

	return @files;
}