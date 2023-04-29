try:
    #여기에 [1, 2, 3] → "List"의 대응관계를 만들어봅시다.
    my_dict = {1:"Integer", 'a':"String", (1,2,3):"Tuple"}
    my_dict[[1,2,3]] = "List"
    
    
except TypeError:
    print("List는 Dictionary의 Key가 될 수 없습니다.")