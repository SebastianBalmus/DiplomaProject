""" from https://github.com/keithito/tacotron """
import re
from text import cleaners
from text.symbols import symbols, symbols_ro


# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

# Mappings from symbol to numeric ID and vice versa for Romanian symbols:
_symbol_to_id_ro = {s: i for i, s in enumerate(symbols_ro)}
_id_to_symbol_ro = {i: s for i, s in enumerate(symbols_ro)}

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')


def text_to_sequence(text, cleaner_names):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

    The text can optionally have ARPAbet sequences enclosed in curly braces embedded
    in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through

    Returns:
      List of integers corresponding to the symbols in the text
  '''
  sequence = []

  use_ro = cleaner_names[0] == "basic_cleaners"

  # Check for curly braces and treat their contents as ARPAbet:
  while len(text):
    m = _curly_re.match(text)
    if not m:
      sequence += _symbols_to_sequence(_clean_text(text, cleaner_names), use_ro)
      break
    sequence += _symbols_to_sequence(_clean_text(m.group(1), cleaner_names), use_ro)
    sequence += _arpabet_to_sequence(m.group(2), use_ro)
    text = m.group(3)

  return sequence


def sequence_to_text(sequence, ro_chars):
  '''Converts a sequence of IDs back to a string'''
  if ro_chars:
    symbol_map = _id_to_symbol_ro
  else:
    symbol_map = _id_to_symbol

  result = ''
  for symbol_id in sequence:
    if symbol_id in symbol_map:
      s = symbol_map[symbol_id]
      # Enclose ARPAbet back in curly braces:
      if len(s) > 1 and s[0] == '@':
        s = '{%s}' % s[1:]
      result += s
  return result.replace('}{', ' ')


def _clean_text(text, cleaner_names):
  for name in cleaner_names:
    cleaner = getattr(cleaners, name)
    if not cleaner:
      raise Exception('Unknown cleaner: %s' % name)
    text = cleaner(text)
  return text


def _symbols_to_sequence(symbols, use_ro=False):
  _symbol_to_id_map = _symbol_to_id_ro if use_ro else _symbol_to_id
  return [_symbol_to_id_map[s] for s in symbols if _should_keep_symbol(s, use_ro)]

def _arpabet_to_sequence(text, use_ro=False):
  return _symbols_to_sequence(['@' + s for s in text.split()], use_ro)

def _should_keep_symbol(s, use_ro=False):
  _symbol_to_id_map = _symbol_to_id_ro if use_ro else _symbol_to_id
  return s in _symbol_to_id_map and s != '_' and s != '~'
