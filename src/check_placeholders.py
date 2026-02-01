"""检查占位符数量"""
import re
from pathlib import Path

content = Path('paper/all_tables.md').read_text(encoding='utf-8')
placeholders = len(re.findall(r'(placeholder|PLACEHOLDER|TODO|TBD|XXX)', content, re.I))
not_avail = content.count('not available')
print(f'Placeholder count: {placeholders}')
print(f'"not available" count: {not_avail}')
print(f'Total suspicious: {placeholders + not_avail}')

# 检查Table 12
if 'Table 12' in content:
    table12_start = content.find('Table 12')
    table12_section = content[table12_start:table12_start+500]
    print(f'\nTable 12 section preview:')
    print(table12_section[:300])
