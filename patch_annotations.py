import glob
patched = []
for f in glob.glob('/ai-toolkit/**/*.py', recursive=True):
    try:
        s = open(f, encoding='utf-8').read()
    except Exception:
        continue
    if 'list[torch.Tensor]' in s:
        s = s.replace('list[torch.Tensor]', 'typing.List[torch.Tensor]')
        if 'import typing' not in s:
            lines = s.split('\n')
            ins = 0
            for i, l in enumerate(lines[:20]):
                if l.startswith('from __future__'):
                    ins = i + 1
            lines.insert(ins, 'import typing')
            s = '\n'.join(lines)
        open(f, 'w', encoding='utf-8').write(s)
        patched.append(f)
print('PATCHED files:', patched)
