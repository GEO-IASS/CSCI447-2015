#!/usr/bin/perl

=for comment
Machine Learning Assignment 1 ARFF Converter
Authors: Clint Cooper, Emily Rohrbough, Samuel Micka
Date: 08/31/15

Takes header and data files to create an ARFF file for use with WEKA

---Begin Header Format---
Title: <String>
Creator: <String>(s)
Donor: <String>(s)
Date: MMM, YYYY
Attributes:
Name:Type

Types include: Numeric, String, (Options), Date Format
----End Header Format----
=cut

use strict;
use warnings;
use diagnostics;
use List::MoreUtils qw(first_index);

print "Please enter folder to scan: ";
my $folder = <>;
my @alphabet = ('a','b','c','d','e','f','g','h','i','j');
chomp($folder);
if ((substr $folder, -1) eq '/'){
	chop($folder); # Remove extra / at end of path
}
$folder =~ tr/\\//; # Remove user escaping spaces.
my $newarff = $folder;
$folder = "Data/" . $folder;

# Open files for reading and writing for this transaction
open(my $header, '<', "$folder/header") or die "Unable to locate header file.";
open(my $data, '<', "$folder/data") or die "Unable to locate data file.";
open(my $output, '>', "$folder/$newarff.arff") or die "Unable to write new arff file";

# Get lines from crafted header file
my @lines;
while(<$header>){
	push (@lines, $_);
}

# Write comments to arff file from content in header file
print $output "% 1. $lines[0]%\n% 2. Sources:\n";
my $i = 0;
my $split = first_index { index($_, 'Attributes:') != -1} @lines;
foreach my $x (@lines[1 .. $split]){
	if (index($x, 'Creator:') != -1) {
		print $output "%\t\t($alphabet[$i]) $x";
		$i++;
	} elsif (index($x, 'Donor:') != -1) {
		print $output "%\t\t($alphabet[$i]) $x";
		$i++;
	} elsif (index($x, 'Date:') != -1) {
		print $output "%\t\t($alphabet[$i]) $x";
		$i++;
	}
}

# Write relation line to new arff file based on folder name
my $relation = substr($folder, index($folder, '/')+1, length $folder);
print $output "%\n\@RELATION $relation\n\n";

# Write attributes to arff file from types in header file. Needs some more work for all types to be supported.
foreach my $y (@lines[$split+1 .. $#lines]){
	$y =~s/ /_/g;
	$y =~s/\'//g;
	print $output "\@ATTRIBUTE\t" . substr($y, 0, index($y, ":")) . "\t";
	if (index($y, 'date') != -1) {
		#print "\#2\n";
		my $out = substr($y, index($y, ":")+1, length $y);
		chomp($out);
		print $output uc $out . "yyyy-MM-dd HH:mm:ss\n";
	} elsif (index($y, '(') == -1 and index($y, '{') == -1) {
		#print "\#1\n";
		my $out = substr($y, index($y, ":")+1, length $y);
		chomp($out);
		print $output uc $out . "\n";
	} else {
		#print "\#3\n";
		my $out = substr($y, index($y, ":")+1, length $y);
		chomp($out);
		$out =~s/[_]+//g;
		$out =~s/[\(]+/\{/g;
		$out =~s/[\)]+/\}/g;
		#print "$out\n";
		print $output $out ."\n";
	}
}

# Write data lines to arff file
print $output "\n\@DATA\n";
@lines = ();
while (<$data>){
	push (@lines, $_);
}
foreach my $z (@lines){
	print $output $z;
}

# Close opened files
close $header;
close $data;
close $output;

# Print finished confirmation
print("Successfully created $folder/$newarff.arff\n");
	


