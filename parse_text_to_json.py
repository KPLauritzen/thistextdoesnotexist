import json
import re
text = ""
i_sample = 0
with open('unconditional.txt', 'r') as f:
    for line in f:
        if line[0] == '=' and line[-2] == '=':
            # This is end of sample
            text += '<br>'
            out = {'output':text}
            with open(f'data/unconditional{i_sample}.json', 'w') as f:
                json.dump(out, f)
            # Get ready for next sample
            i_sample += 1
            text = ""
        else:
            if line == '\n': line = '<br>'
            if '<|endoftext|>' in line:
                # replace endoftext with linebreaks
                line = line.replace('<|endoftext|>', '<br><br><|endoftext|><br><br>')
            text += line + ' '
