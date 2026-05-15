# By Amos Satterlee
# Improved by MS Copilot AI

def hello():
    print("Hello!")

def area(width, height):
    return width * height

def print_welcome(name):
    print("Welcome,", name)

# Ask for user's name
name = input("Your Name: ")
hello()
print_welcome(name)

print("\nTo find the area of a rectangle,")
print("Enter the width and height below.\n")

# Get width
w = float(input("Width: "))
while w <= 0:
    print("Must be a positive number")
    w = float(input("Width: "))

# Get height
h = float(input("Height: "))
while h <= 0:
    print("Must be a positive number")
    h = float(input("Height: "))

print(f"Width = {w}, Height = {h}, so Area = {area(w, h)}")

