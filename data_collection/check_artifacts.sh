#!/bin/bash
# Check wiki artifacts in word list
# Usage: ./check_artifacts.sh [input_file]

INPUT="${1:-validated_data/words_needs_review.txt}"

echo "========================================"
echo "WIKI ARTIFACT CHECK"
echo "========================================"
echo "File: $INPUT"
echo ""

# Count special prefixes
echo "PREFIX COUNTS:"
echo "----------------------------------------"
for char in '*' '>' '|' '/' '_' '%' '#' '&' '@' '!' '=' '-'; do
    count=$(grep -c "^$char" "$INPUT" 2>/dev/null || echo "0")
    printf "  '%s' prefix: %s\n" "$char" "$count"
done

# Check broken anusvara
echo ""
echo "BROKEN ANUSVARA (ം at start):"
echo "----------------------------------------"
anusvara_count=$(grep -c "^ം" "$INPUT" 2>/dev/null || echo "0")
echo "  Count: $anusvara_count"

# Show samples
echo ""
echo "SAMPLE ARTIFACTS:"
echo "----------------------------------------"
echo "  Words starting with '*':"
grep '^\*' "$INPUT" 2>/dev/null | head -3 | sed 's/^/    /'
echo ""
echo "  Words starting with '>':"
grep '^>' "$INPUT" 2>/dev/null | head -3 | sed 's/^/    /'
echo ""
echo "  Words starting with '|':"
grep '^|' "$INPUT" 2>/dev/null | head -3 | sed 's/^/    /'
echo ""
echo "  Words starting with 'ം' (broken):"
grep '^ം' "$INPUT" 2>/dev/null | head -3 | sed 's/^/    /'

# Total
echo ""
echo "========================================"
total=$(grep -E '^[\*>|/%_#&@!=-]|^ം' "$INPUT" 2>/dev/null | wc -l)
echo "TOTAL ARTIFACTS: $total"
echo "========================================"
