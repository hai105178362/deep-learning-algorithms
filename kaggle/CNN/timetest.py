import datetime

d = datetime.datetime.today()
epoch = 5
record = "{}-{}-{}-e{}".format(d.day, d.hour, d.minute, epoch)
print(record)
print(d.day)
print(d.hour)
print(d.minute)
