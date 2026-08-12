#!/usr/bin/bash
# convert data ($1 - input file, $2 - amount of locally scaled white noise to add to C and P
noise=$2
if [ -z $2 ]
then
    noise=0
fi
#log of concentrations - log($20), pressures - log($16)
rm res_pr.txt res_pr_P.txt res_pr_C.txt
cat $1 | awk "BEGIN{srand();}{ if (\$1 == \"u_inp\") {printf \"\n%g,%g,%g,%g,%g,\",\$2,\$4,\$6,\$8,\$10}; \
if (\$1 == \"a\") {if ((\$10 == \"0.001\") && (\$12 == \"0.001\") && (\$14 ==\"0.1\")) {printf \"%g,%g,%g,%g,%g,%g,%g,\",\$2,\$4,\$6,\$8,\$12,\$14,log(\$20*(1.0+(2.0*rand()-1.0)*$noise))} else\
{printf \"%g,%g,%g,\",\$12,\$14,log(\$20*(1.0+(2.0*rand()-1.0)*$noise))}}}" | sed 's/,$//' >> res_pr_C.txt

cat $1 | awk "BEGIN{srand();}{ if (\$1 == \"u_inp\") {printf \"\n%g,%g,%g,%g,%g,\",\$2,\$4,\$6,\$8,\$10}; \
if (\$1 == \"a\") {if ((\$10 == \"0.001\") && (\$12 == \"0.001\") && (\$14 ==\"0.1\")) {printf \"%g,%g,%g,%g,%g,%g,%g,\",\$2,\$4,\$6,\$8,\$12,\$14,log(\$16*(1.0+(2.0*rand()-1.0)*$noise))} else\
{printf \"%g,%g,%g,\",\$12,\$14,log(\$16*(1.0+(2.0*rand()-1.0)*$noise))}}}" | sed 's/,$//' >> res_pr_P.txt

# remove rows with "nan" in either of files
paste "res_pr_C.txt" "res_pr_P.txt" | \
awk '
  !(($0 ~ /nan/)||($0 ~ /inf/)) {
    print $1 > f1out
    print $2 > f2out
  }
' f1out="res_pr_C_clean.txt" f2out="res_pr_P_clean.txt"

rm res_pr_C.txt res_pr_P.txt
mv res_pr_C_clean.txt res_pr_C.txt
mv res_pr_P_clean.txt res_pr_P.txt

awk -F',' '
NR==2 {
    n = NF
    for (i=1; i<=n; i++) {
        printf "x%d%s", i, (i<n ? "," : "\n")
    }
}
NR>=2 { print }
' res_pr_C.txt > tmp.csv && mv tmp.csv res_pr_C.txt

awk -F',' '
NR==2 {
    n = NF
    for (i=1; i<=n; i++) {
        printf "x%d%s", i, (i<n ? "," : "\n")
    }
}
NR>=2 { print }
' res_pr_P.txt > tmp.csv && mv tmp.csv res_pr_P.txt

ln -s res_pr_C.txt res_pr.txt
