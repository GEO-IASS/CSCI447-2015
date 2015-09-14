#!/usr/bin/perl

=for comment
Machine Learning Assignment 1 Results to CSV converter
Authors: Clint Cooper, Emily Rohrbough
Date: 09/13/15

Takes the returned .out files and extracts useful information.

=cut

#use strict;
#use warnings;
#use diagnostics;
use List::Util qw( min max );
use List::MoreUtils qw(first_index);
use List::MoreUtils qw(uniq);
use Data::Dumper qw(Dumper);
use Math::Round;
 

#print "Please enter folder to convert: ";
#my $folder = <>;
#chomp($folder);
#if ((substr $folder, -1) eq '/'){
#	chop($folder); # Remove extra / at end of path
#}
#$folder =~ tr/\\//; # Remove user escaping spaces.
my $folder = "Results";

open(my $output, '>', "results.csv") or die "Unable to write new data file";

my @list = process_files($folder);

my @correct;
my @log;
my @square;
my @hinge;
my @total;
my @dataset;
my @algorithm;
my $worcor;
my $wortot;
my $worlog;
my $worsqr;
my $worhin;
my @midhin;

foreach my $file (@list){
	#print $file, "\n";
	$worcor = 0;
	$wortot = 0;
	$worlog = 0;
	$worsqr = 0;
	$worhin = 0;
	if (-f $file) {
		open(INFILE, $file) or die "Cannot open $_!.\n";
		$file =~ s/Results\///g;
		my @title = split('-', $file);

		if ($title[-1] =~ m/^results.out/) {
			while(my $line = <INFILE>) {
				if ($line =~ /^Correctly Classified Instances/){
					#print "class\n";
					$line =~ s/^Correctly Classified Instances\s*//g;
					$line =~ s/(\d+).*$/$1/;
					$worcor = $line;
				} elsif ($line =~ /^Total Number of Instances/){
					#print "total\n";
					$line =~ s/^Total Number of Instances\s*//g;
					$line =~ s/(\d+).*$/$1/;
					$wortot = $line;
				} elsif ($line =~ /^Class complexity | scheme/){
					#print "log\n";
					$line =~ s/^Class complexity | scheme\s*//g;
					$line =~ s/\|(\d+.\d+).*$/$1/;
					$worlog = $line;
				} elsif ($line =~ /^Root mean squared error/){
					#print "square\n";
					$line =~ s/^Root mean squared error\s*//g;
					$worsqr = $line ** 2;
				} 
			}
			if ($#title == 2){
				push @dataset, ucfirst($title[0]);
			} elsif ($#title == 3){
				push @dataset, join('-', ucfirst($title[0]), ucfirst($title[1]));
			} elsif ($#title == 4){
				push @dataset, join('-', ucfirst($title[0]), ucfirst($title[1]), ucfirst($title[2]));
			}
			push @algorithm, $title[-2];
			push @log, $worlog;
			push @total, $wortot;
			push @correct, $worcor;
			push @square, $worsqr;
		} else {
			$worhin = 0;
			my $min = 1;
			while(my $line = <INFILE>) {
				if ($line =~ m/\s+\d+/) {
					my $wrong = index($line, '  +  ');
					#print $wrong;
					$line =~ s/^.{35}//g;
					$line =~ s/^(\d{\W\d+}).*$/$1/g;
					if ($line < $min and $line > 0){
						$min = $line;
					}
					$line = $line * -($wrong/abs($wrong));
					chomp($line);
					push @midhin, $line;
				}
			}
			#print $min;
			foreach my $x (@midhin){
				$worhin += $x/$min;
			}
			$worhin = $worhin/(length midhin);
			#print $worhin;
			chomp($worhin);
			push @hinge, $worhin;
		}
	}
}
#print "\n";

chomp(@dataset);
chomp(@algorithm);
chomp(@correct);
chomp(@total);
chomp(@log);
chomp(@square);
chomp(@hinge);

print $output "Data Set,Algorithm,# Correct,# Total,Log Loss,Square Loss,Hinge Loss\n";
for my $i (0 .. $#dataset) {
	print $output $dataset[$i],',',uc($algorithm[$i]),',',$correct[$i],',',$total[$i],',',nearest(0.01, $hinge[$i]),',',nearest(0.01, $square[$i]),',',nearest(0.01, $log[$i]),"\n";
}


sub process_files {    
	my $path = shift;
	opendir (DIR, $path) or die "Unable to open $path: $!";
	my @files =
		map { $path . '/' . $_ }
		grep { !/^\.{1,2}$/ } # Don't climb up
		grep { !/^\.\w+$/ } # Remove hidden files
		grep { !/^\w+.txt/ } # Ignores .txt files
		grep { !/^\w+.log/ } # Ignores .log files
		grep { !/^\w+.out/ } # Ignores. out files
		# At this point, this has turned into a .gitignore, then line count.
		readdir (DIR);

	closedir (DIR);

	for (@files) {
		if (-d $_) {
			push @files, process_files ($_);
		}
	}
	return @files;
}



