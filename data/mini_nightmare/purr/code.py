
class Cat:
    def __init__(self, name):
        self.name = name
        self.hunger = 0

    def purr(self):
        return self.name + ": Purr"

    def meow(self):
        return self.name + ": Meow!"

    def intense_meow(self):
        return self.name + ": MEOW!!!"

    def feed(self, food):
        if food == "fish":
            self.hunger -= 3
        elif food == "meat":
            self.hunger -= 6
    
    def play(self):
        self.hunger += 5

    def step(self):
        result = None
        if self.hunger > 10:
            result = self.meow()
            self.feed("fish")
        elif self.hunger > 20:
            result = self.intense_meow()
            self.feed("meat")
        else:
            result = self.purr()
        self.play()
        return result


def cat_status(name, k):

    time_step = 0
    cat = Cat(name)

    while True:
        result = cat.step()
        if time_step == k:
            return result
        time_step += 1


for i in range(10):
    print(cat_status("Nono", i))