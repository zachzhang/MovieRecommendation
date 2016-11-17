t = "We've\xe5\xcabeen invited to attend TEDxTeen, an independently organized TED event focused on encouraging youth to find \x89\xdb\xcfsimply irresistible\x89\xdb\x9d solutions to the complex issues we face every day.,"

t2 = t.decode('unicode_escape').encode('ascii', 'ignore').strip()
#import sys
#sys.stdout.write(t2.strip('\n\r'))

print t
print t2.strip('\n\r')

print t2