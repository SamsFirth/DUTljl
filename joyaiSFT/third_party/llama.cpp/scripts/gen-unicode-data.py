import regex
import ctypes
import unicodedata

class CoodepointFlags(ctypes.Structure):
    _fields_ = [('is_undefined', ctypes.c_uint16, 1), ('is_number', ctypes.c_uint16, 1), ('is_letter', ctypes.c_uint16, 1), ('is_separator', ctypes.c_uint16, 1), ('is_accent_mark', ctypes.c_uint16, 1), ('is_punctuation', ctypes.c_uint16, 1), ('is_symbol', ctypes.c_uint16, 1), ('is_control', ctypes.c_uint16, 1)]
assert ctypes.sizeof(CoodepointFlags) == 2
MAX_CODEPOINTS = 1114112
regex_number = regex.compile('\\p{N}')
regex_letter = regex.compile('\\p{L}')
regex_separator = regex.compile('\\p{Z}')
regex_accent_mark = regex.compile('\\p{M}')
regex_punctuation = regex.compile('\\p{P}')
regex_symbol = regex.compile('\\p{S}')
regex_control = regex.compile('\\p{C}')
regex_whitespace = regex.compile('\\s')
codepoint_flags = (CoodepointFlags * MAX_CODEPOINTS)()
table_whitespace = []
table_lowercase = []
table_uppercase = []
table_nfd = []
for codepoint in range(MAX_CODEPOINTS):
    char = chr(codepoint)
    flags = codepoint_flags[codepoint]
    flags.is_number = bool(regex_number.match(char))
    flags.is_letter = bool(regex_letter.match(char))
    flags.is_separator = bool(regex_separator.match(char))
    flags.is_accent_mark = bool(regex_accent_mark.match(char))
    flags.is_punctuation = bool(regex_punctuation.match(char))
    flags.is_symbol = bool(regex_symbol.match(char))
    flags.is_control = bool(regex_control.match(char))
    flags.is_undefined = bytes(flags)[0] == 0
    assert not flags.is_undefined
    if bool(regex_whitespace.match(char)):
        table_whitespace.append(codepoint)
    lower = ord(char.lower()[0])
    if codepoint != lower:
        table_lowercase.append((codepoint, lower))
    upper = ord(char.upper()[0])
    if codepoint != upper:
        table_uppercase.append((codepoint, upper))
    norm = ord(unicodedata.normalize('NFD', char)[0])
    if codepoint != norm:
        table_nfd.append((codepoint, norm))
ranges_flags = [(0, codepoint_flags[0])]
for codepoint, flags in enumerate(codepoint_flags):
    if bytes(flags) != bytes(ranges_flags[-1][1]):
        ranges_flags.append((codepoint, flags))
ranges_flags.append((MAX_CODEPOINTS, CoodepointFlags()))
ranges_nfd = [(0, 0, 0)]
for codepoint, norm in table_nfd:
    start = ranges_nfd[-1][0]
    if ranges_nfd[-1] != (start, codepoint - 1, norm):
        ranges_nfd.append(None)
        start = codepoint
    ranges_nfd[-1] = (start, codepoint, norm)

def out(line=''):
    print(line, end='\n')
out('// generated with scripts/gen-unicode-data.py\n\n#include "unicode-data.h"\n\n#include <cstdint>\n#include <vector>\n#include <unordered_map>\n#include <unordered_set>\n')
out('const std::vector<std::pair<uint32_t, uint16_t>> unicode_ranges_flags = {  // start, flags // last=next_start-1')
for codepoint, flags in ranges_flags:
    flags = int.from_bytes(bytes(flags), 'little')
    out('{0x%06X, 0x%04X},' % (codepoint, flags))
out('};\n')
out('const std::unordered_set<uint32_t> unicode_set_whitespace = {')
out(', '.join(('0x%06X' % cpt for cpt in table_whitespace)))
out('};\n')
out('const std::unordered_map<uint32_t, uint32_t> unicode_map_lowercase = {')
for tuple in table_lowercase:
    out('{0x%06X, 0x%06X},' % tuple)
out('};\n')
out('const std::unordered_map<uint32_t, uint32_t> unicode_map_uppercase = {')
for tuple in table_uppercase:
    out('{0x%06X, 0x%06X},' % tuple)
out('};\n')
out('const std::vector<range_nfd> unicode_ranges_nfd = {  // start, last, nfd')
for triple in ranges_nfd:
    out('{0x%06X, 0x%06X, 0x%06X},' % triple)
out('};\n')