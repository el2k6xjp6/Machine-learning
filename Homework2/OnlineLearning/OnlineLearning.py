import math

class OnlineLearning:
    def __init__(self,file,a,b):
        self.file=file
        self.a=a
        self.b=b

    def __ReadFile(self):
        trail = []
        with open(self.file, 'r') as File:
            for line in File:
                line = line.strip()
                if line != '':
                    trail.append(line)
        return trail

    def __BetaBinomailConjugation(self,trail):
        count=0
        for batch in trail:
            count+=1
            print('='*35)
            print("case {}: {}".format(count,batch))
            (zero,one)=(batch.count('0'),batch.count('1'))
            p=one/(zero+one)
            print('Likelihood: {}'.format((math.factorial(one+zero)*(p**one)*(1-p)**zero)/(math.factorial(one)*math.factorial(zero))))
            print('Beta prior:     {} {}'.format(self.a,self.b))
            self.a+=one
            self.b+=zero
            print('Beta posterior: {} {}'.format(self.a,self.b))

    def run(self):
        trail=self.__ReadFile()
        self.__BetaBinomailConjugation(trail)