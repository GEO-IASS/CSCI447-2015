#!/usr/bin/perl

=for comment
Solution for Easy Challenge #002
Author: Keydrain
Date: 03/25/15

A simple calculator for number of lines of code contained within a project.
=cut

use strict;
use warnings;
use diagnostics;
use Data::Dumper;

print "Please enter folder to scan: ";
my $folder = <>;
chomp($folder);
if ((substr $folder, -1) eq '/'){
	chop($folder); # Remove extra / at end of path
}
$folder =~ tr/\\//; # Remove user escaping spaces.

my @list = process_files($folder);
#print Dumper \@list;

my $counter = 0;
my $total = 0;

foreach my $file (@list){
	if (-f $file) {
		open(INFILE, $file) or die "Cannot open $_!.\n";

		# This loops through each line of the file
		while(my $line = <INFILE>) {
			if (index($line, ',,') != -1){
				print $file . "\n";
			}
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