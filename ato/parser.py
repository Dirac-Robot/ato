def parse_command(command):
    tokens = []
    i = 0
    length = len(command)
    while i < length:
        while i < length and command[i].isspace():
            i += 1
        if i >= length:
            break
        start = i
        while i < length and not command[i].isspace() and command[i] not in ('=', ':'):
            i += 1
        if i < length and command[i] == ':' and i+1 < length and command[i+1] == '=':
            key = command[start:i]
            i += 2
            if i < length:
                value, i = parse_colon_equals_string(command, i)
            else:
                value = ''
            tokens.append(f'{key}:={value}')
        elif i < length and command[i] == '=':
            key = command[start:i]
            i += 1
            if i < length:
                value, i = parse_value(command, i)
            else:
                value = ''
            tokens.append(f'{key}={value}')
        else:
            if command[start] in ['[', '(', '{']:
                token, i = parse_bracketed_value(command, start)
                tokens.append(token)
            else:
                token_start = start
                while i < length and not command[i].isspace():
                    i += 1
                tokens.append(command[token_start:i])
    return tokens


def parse_value(command, i):
    if i < len(command):
        if command[i] in ['[', '(', '{']:
            return parse_bracketed_value(command, i)
        else:
            start = i
            while i < len(command) and not command[i].isspace():
                i += 1
            return command[start:i], i
    return '', i


def parse_colon_equals_string(command, i):
    length = len(command)
    if i >= length:
        return '', i
    if command[i] in ('"', "'"):
        return parse_quoted_string(command, i)
    start = i
    while i < length and not command[i].isspace():
        i += 1
    return command[start:i], i


def parse_quoted_string(command, i):
    quote_char = command[i]
    i += 1
    value = []
    length = len(command)
    while i < length:
        c = command[i]
        if c == '\\' and i+1 < length:
            next_char = command[i+1]
            if next_char in (quote_char, '\\'):
                value.append(next_char)
                i += 2
            else:
                value.append(c)
                i += 1
        elif c == quote_char:
            i += 1
            break
        else:
            value.append(c)
            i += 1
    return ''.join(value), i


def parse_bracketed_value(command, i):
    brackets = {'[': ']', '{': '}', '(': ')'}
    opening_bracket = command[i]
    closing_bracket = brackets[opening_bracket]
    value = [opening_bracket]
    i += 1
    length = len(command)
    stack = [closing_bracket]
    while i < length and stack:
        c = command[i]
        if c == '\\' and i+1 < length:
            value.append(c)
            value.append(command[i+1])
            i += 2
        elif c in brackets:
            stack.append(brackets[c])
            value.append(c)
            i += 1
        elif c == stack[-1]:
            stack.pop()
            value.append(c)
            i += 1
        elif c in ('"', "'"):
            quoted, i = parse_quoted_string_with_quotes(command, i)
            value.append(quoted)
        else:
            value.append(c)
            i += 1
    return ''.join(value), i


def parse_quoted_string_with_quotes(command, i):
    quote_char = command[i]
    value = [quote_char]
    i += 1
    length = len(command)
    while i < length:
        c = command[i]
        if c == '\\' and i+1 < length:
            value.append(c)
            value.append(command[i+1])
            i += 2
        elif c == quote_char:
            value.append(c)
            i += 1
            break
        else:
            value.append(c)
            i += 1
    return ''.join(value), i
