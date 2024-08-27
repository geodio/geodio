from core.cell import Variable, Constant


def main():
    var0 = Variable(0)
    var1 = Variable(1)
    c_4 = Constant(4)
    func = (var0 + c_4) / var1
    print(func([10, 2]))


if __name__ == '__main__':
    main()
