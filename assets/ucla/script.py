import os
b = []
for file in os.listdir('.'):
	tail = file.split("-")[-1]
	number = tail.split(".")[0]
	nf = number.replace("0", "")
	os.rename(
		f"./{file}", 
		f"./{nf}.jpg")
print(b)
