#!/bin/bash
# ============================================================================
# MORNINGSTAR — Benchmark Suite
# Testet Morningstar Modelle gegen Coding-Challenges
# Developed by: Ali Nasser (github.com/morningstarnasser)
# ============================================================================
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

MODEL="${1:-morningstar}"
PASS=0
FAIL=0
TOTAL=0

echo ""
echo -e "${CYAN}${BOLD}"
echo "  ╔══════════════════════════════════════════════════╗"
echo "  ║    MORNINGSTAR — Benchmark Suite                 ║"
echo "  ║    by Ali Nasser (github/morningstarnasser)      ║"
echo "  ╚══════════════════════════════════════════════════╝"
echo -e "${NC}"
echo -e "  Model: ${BOLD}${MODEL}${NC}"
echo ""

# ─── Helper ─────────────────────────────────────────────
run_test() {
  local name="$1"
  local prompt="$2"
  local check="$3"
  local timeout="${4:-30}"

  TOTAL=$((TOTAL + 1))
  echo -n -e "  [${TOTAL}] ${name}... "

  RESPONSE=$(timeout "${timeout}" ollama run "$MODEL" "$prompt" 2>/dev/null || echo "TIMEOUT")

  if echo "$RESPONSE" | grep -qiE "$check"; then
    echo -e "${GREEN}PASS${NC}"
    PASS=$((PASS + 1))
  else
    echo -e "${RED}FAIL${NC}"
    FAIL=$((FAIL + 1))
    if [ "${VERBOSE:-0}" = "1" ]; then
      echo -e "    ${YELLOW}Response: ${RESPONSE:0:200}${NC}"
    fi
  fi
}

# ─── Tests ──────────────────────────────────────────────

echo -e "${BOLD}  Python${NC}"
run_test "FizzBuzz" \
  "Write a Python function fizzbuzz(n) that returns 'Fizz' for multiples of 3, 'Buzz' for multiples of 5, 'FizzBuzz' for both, else the number. Only the function, no explanation." \
  "def fizzbuzz|FizzBuzz"

run_test "Reverse String" \
  "Write a Python one-liner that reverses a string s. Only the code." \
  "reverse|\\[::-1\\]"

run_test "List Comprehension" \
  "Write a Python one-liner to get all even numbers from a list nums. Only the code." \
  "for.*in.*nums|filter|%.*2.*==.*0"

run_test "Fibonacci" \
  "Write a Python function fibonacci(n) that returns the nth Fibonacci number. Only the function." \
  "def fibonacci"

run_test "Binary Search" \
  "Write a Python function binary_search(arr, target) that returns the index. Only the function." \
  "def binary_search|mid|left|right|lo|hi"

echo ""
echo -e "${BOLD}  JavaScript${NC}"
run_test "Arrow Function" \
  "Write a JavaScript arrow function that filters an array to only even numbers. Only the code." \
  "=>|filter|%.*2"

run_test "Promise" \
  "Write a JavaScript function that returns a Promise resolving after n milliseconds. Only the code." \
  "Promise|resolve|setTimeout"

run_test "Destructuring" \
  "Write JavaScript to destructure name and age from an object person. Only the code." \
  "const.*{|let.*{|name.*age|age.*name"

echo ""
echo -e "${BOLD}  TypeScript${NC}"
run_test "Generic Function" \
  "Write a TypeScript generic function identity<T>(arg: T): T. Only the code." \
  "function.*<T>|<T>.*=>|:.*T"

run_test "Interface" \
  "Write a TypeScript interface User with id (number), name (string), email (string). Only the code." \
  "interface User|type User"

echo ""
echo -e "${BOLD}  SQL${NC}"
run_test "SELECT with JOIN" \
  "Write a SQL query to get all orders with customer names using a JOIN. Only the SQL." \
  "SELECT|JOIN|FROM.*orders"

run_test "GROUP BY" \
  "Write SQL to count orders per customer, ordered by count descending. Only the SQL." \
  "GROUP BY|COUNT|ORDER BY"

echo ""
echo -e "${BOLD}  Bash${NC}"
run_test "File Count" \
  "Write a bash one-liner to count all .py files in current directory recursively. Only the command." \
  "find|wc|\\*\\.py|ls"

echo ""
echo -e "${BOLD}  Algorithmen${NC}"
run_test "Sorting" \
  "Write a Python quicksort implementation. Only the function." \
  "def.*sort|pivot|partition|quick"

run_test "Tree Traversal" \
  "Write a Python function for in-order traversal of a binary tree. Only the function." \
  "def.*inorder|def.*traverse|left|right"

echo ""
echo -e "${BOLD}  Security${NC}"
run_test "SQL Injection Aware" \
  "How would you safely query a database in Python with user input? Show code." \
  "parameterized|placeholder|%s|\?|execute.*,|bind|sanitize"

run_test "Password Hashing" \
  "Show how to hash a password in Python. Only the code." \
  "bcrypt|argon2|hashlib|pbkdf2|scrypt"

echo ""
echo -e "${BOLD}  Error Handling${NC}"
run_test "Try/Except" \
  "Write a Python function that reads a JSON file safely with error handling. Only the function." \
  "try|except|json|open"

run_test "TypeScript Error" \
  "Write a TypeScript function that fetches a URL and handles errors. Only the function." \
  "try|catch|fetch|async|Error"

# ─── Results ────────────────────────────────────────────
echo ""
echo -e "${BOLD}  ════════════════════════════════════════════${NC}"
SCORE=$((PASS * 100 / TOTAL))

if [ $SCORE -ge 90 ]; then
  COLOR=$GREEN
  GRADE="S"
elif [ $SCORE -ge 80 ]; then
  COLOR=$GREEN
  GRADE="A"
elif [ $SCORE -ge 70 ]; then
  COLOR=$YELLOW
  GRADE="B"
elif [ $SCORE -ge 60 ]; then
  COLOR=$YELLOW
  GRADE="C"
else
  COLOR=$RED
  GRADE="F"
fi

echo -e "  ${COLOR}${BOLD}  Model:  ${MODEL}${NC}"
echo -e "  ${COLOR}${BOLD}  Score:  ${PASS}/${TOTAL} (${SCORE}%) — Grade ${GRADE}${NC}"
echo -e "  ${GREEN}  Pass:   ${PASS}${NC}"
echo -e "  ${RED}  Fail:   ${FAIL}${NC}"
echo -e "${BOLD}  ════════════════════════════════════════════${NC}"
echo ""
echo -e "  Vergleich: ${CYAN}VERBOSE=1 ./benchmark.sh morningstar-32b${NC}"
echo ""
