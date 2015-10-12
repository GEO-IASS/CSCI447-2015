#!/usr/bin/perl

=for comment
Machine Learning Assignment 2 Log File Parser
Authors: Clint Cooper, Emily Rohrbough, Leah Thompson
Date: 10/10/15

Takes the generated .log files that contain the results of the tests 
on the NN and prints the data organized to a tsv file.
=cut

#use strict;
#use warnings;
#use diagnostics;

my $folder = "Results";
my @actSets = ("[['']]", "[['L', 'L', 'L']]", "[['S', 'S', 'S']]", "[['L', 'L', 'L'], ['L', 'L']]", "[['S', 'S', 'S'], ['S', 'S']]", "[['G', 'G', 'G']]", "[['G', 'G', 'G', 'G', 'G']]", "[['G', 'G', 'G', 'G', 'G', 'G', 'G']]");
my @funcSets = ("['L']", "['S']", "['R']");

open(my $output, '>', "results.csv") or die "Unable to write new data file";
my @list = process_files($folder);

foreach my $file (@list){
	if (-f $file) {
		open(INFILE, $file) or die "Cannot open $_!.\n";
		$file =~ s/$folder\///;
		$file =~ s/RBF//;
		$file =~ s/Results.log//;
		$file =~ s/\-\d//;
		$file =~ s/\-//;
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

		my $actIndex = 0;
		++$actIndex until $actSets[$actIndex] eq $activationSet or $actIndex > $#actSets;
		my $funcIndex = 0;
		++$funcIndex until $funcSets[$funcIndex] eq $activationOutput or $funcIndex > $#funcSets;
		if ($funcIndex == 0) { # Linear
			if ($actIndex == 0) { # 0 Layers
				printf $output "L, %d, 0, ", $file;					
			} elsif ($actIndex == 1 or $actIndex == 2) { # 1 Layer
				printf $output "L, %d, 1, ", $file;
			} elsif ($actIndex == 3 or $actIndex == 4) { # 2 Layers
				printf $output "L, %d, 2, ", $file;
			}
		} elsif ($funcIndex == 1) { # Sigmoid
			if ($actIndex == 0) { # 0 Layers
				printf $output "S, %d, 0, ", $file;
			} elsif ($actIndex == 1 or $actIndex == 2) { # 1 Layer
				printf $output "S, %d, 1, ", $file;
			} elsif ($actIndex == 3 or $actIndex == 4) { # 2 Layers
				printf $output "S, %d, 2, ", $file;
			}
		} elsif ($funcIndex == 2) { # Gaussian
			if ($actIndex == 5) { # 3 Nodes
				printf $output "G, %d, 3, ", $file;
			} elsif ($actIndex == 1 or $actIndex == 6) { # 5 Nodes
				printf $output "G, %d, 5, ", $file;
			} elsif ($actIndex == 3 or $actIndex == 7) { # 7 Nodes
				printf $output "G, %d, 7, ", $file;
			}
		}

		my $counter = 0;
		while($line = <INFILE> and $counter < 4) {
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
			printf $output "%.1f, ", $error * 100;
			$counter = $counter + 1;
		}
		printf $output "\n";
	}
}
sub process_files {    
	my $path = shift;
	opendir (DIR, $path) or die "Unable to open $path: $!";
	my @files =
		map { $path . '/' . $_ }
		grep { !/^\.{1,2}$/ } # Don't climb up
		grep { !/^\.\w+$/ }
		readdir (DIR);

	closedir (DIR);

	return @files;
}