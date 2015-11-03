#!/usr/bin/perl

=for comment
Artificial Characters Collector and Standardizer
Author: Clint Cooper
Date: 09/10/15
=cut

use strict;
use warnings;
use diagnostics;

print "Please enter folder to convert: ";
my $folder = <>;
chomp($folder);
if ((substr $folder, -1) eq '/'){
	chop($folder); # Remove extra / at end of path
}
$folder =~ tr/\\//; # Remove user escaping spaces.

open(my $output, '>', "$folder.csv") or die "Unable to write new data file";

my @list = process_files($folder);

my $counter = 0;
my $total = 0;
my @class = ('A','C','D','E','F','G','H','L','P','R');

foreach my $file (@list){
	if (-f $file) {
		open(INFILE, $file) or die "Cannot open $_!.\n";

		# This loops through each line of each file, does some string replace and outputs to csv
		while(my $line = <INFILE>) {
			$line =~s/[\ ]+/\,/g;
			substr($line,0,1,$class[int(substr($line,0,1))-1]);
			$line =~s/A0/R/g;
			print $output $line;
		}
	}
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