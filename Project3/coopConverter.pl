#!/usr/bin/perl

=for comment
Machine Learning Assignment 3 COOP Converter
Authors: Clint Cooper, Emily Rohrbough, Leah Thompson
Date: 11/01/15

Takes header and data files to create an COOP file for use with our NN Testing

---Begin Header Format---
Title: <String>
Creator: <String>(s)
Donor: <String>(s)
Date: MMM, YYYY
Attributes:
Name:Type

Types include: Numeric, String, (Options), Date Format
----End Header Format----

Code can be run by typing in the name of the folder that contains a header and data file.
The header file needs to have the parameters above while the data should be csv or similar.
Warning: Not all possible layouts and conditions have been tested. 
=cut

use strict;
use warnings;
use diagnostics;
use List::MoreUtils qw(first_index);
use List::MoreUtils qw(firstidx);

#print "Please enter folder to scan: ";
#my $folder = <>;
#my @alphabet = ('a','b','c','d','e','f','g','h','i','j');
#chomp($folder);
#if ((substr $folder, -1) eq '/'){
#	chop($folder); # Remove extra / at end of path
#}
#$folder =~ tr/\\//; # Remove user escaping spaces.
my $folder = "tic-tac-toe";
my $newcoop = $folder;
$folder = "Data Sets/" . $folder;

# Open files for reading and writing for this transaction
open(my $header, '<', "$folder/header") or die "Unable to locate header file.";
open(my $data, '<', "$folder/data") or die "Unable to locate data file.";
open(my $output, '>', "Data Sets/$newcoop.coop") or die "Unable to write new coop file";

# Get lines from crafted header file
my @lines;
while(<$header>){
	push (@lines, $_);
}

my $split = first_index { index($_, 'Attributes:') != -1} @lines;

# Write relation line to new coop file based on folder name
my $relation = substr($folder, index($folder, '/')+1, length $folder);

# Write attributes to coop file from types in header file. Needs some more work for all types to be supported.
print $output "\@ATTRIBUTES\n";

my @inputVector;
my @outputVector;
my @collectionVectors;
my $classCol = 0;
my $classCounter = 0;
foreach my $y (@lines[$split+1 .. $#lines]){
	$y =~s/ /_/g;
	$y =~s/\'//g;

	if (index(uc $y, 'STRING') != -1) {
		print $output substr($y, 0, index($y, ":")) . "\n";
		push (@inputVector, 'S');
		push (@collectionVectors, '-');
	} elsif (index(uc $y, 'NUMERIC') != -1) {
		print $output substr($y, 0, index($y, ":")) . "\n";
		push (@inputVector, '#');
		push (@collectionVectors, '-');
	} elsif (index(uc $y, 'CLASS') != -1) {
		$classCol = $classCounter;
		chop($y);
		print $output substr($y, index($y, ":") + 1, length $y) . "\n";
		my $count = 0;
		chop($y);
		for my $el (split /,/, substr($y, index($y, ":")+2, length $y)) {
			#push (@outputVector, $count);
			push (@outputVector, "$el");
			$count++;
		}
	} else { # This is if it is a set of strings...
		my @temp;
		print $output substr($y, 0, index($y, ":")) . "\n";
		chop($y);
		chop($y);
		for my $el (split /,/, substr($y, index($y, "{")+1, length $y)) {
			push (@temp, "$el");
		}
		push (@collectionVectors, \@temp);
		push (@inputVector, 'C');
	}
	$classCounter++;
}

print $output "\nInput: " . join(",", @inputVector) . "\n"; # This is the mask for inputs
print $output "Output: " . join(",", @outputVector) . "\n"; # This is the mask for outputs
print $output "ClassCol: " . $classCol . "\n";

# Write data lines to coop file
print $output "\n\@DATA\n";
@lines = ();
while (<$data>){
	push (@lines, $_);
}
foreach my $r (@lines){
	my $posCounter = 0;
	my $classFound = 0;
	print $output "[";
	chop($r);
	for my $el (split /,/, $r) {
		#print $output '|>' . $el . '<|';
		if ($posCounter == $classCol and $classFound == 0) {
			if ($posCounter != 0) {
				print $output "][";
			}
			print $output (firstidx { $_ eq $el } @outputVector);
			if ($posCounter == 0) {
				print $output "][";
			}
			$classFound = 1;
			$posCounter--;
		} else {
			if ($inputVector[$posCounter] eq 'S') {
				# What if input is a string?
			} elsif ($inputVector[$posCounter] eq '#') {
				chomp($el);
				if ($posCounter >= $#inputVector) {
					print $output $el;
				} else {
					print $output $el . ',';
				}
			} elsif ($inputVector[$posCounter] eq 'C') {
				print $output @{$collectionVectors[$posCounter]};
				print $output (firstidx { $_ eq $el } @{$collectionVectors[$posCounter]});
			} else {
				# Others?
			}
		}
		$posCounter++;
	}
	print $output "]\n";
}

# Close opened files
close $header;
close $data;
close $output;

# Print finished confirmation
print("Successfully created $folder/$newcoop.coop\n");
	


