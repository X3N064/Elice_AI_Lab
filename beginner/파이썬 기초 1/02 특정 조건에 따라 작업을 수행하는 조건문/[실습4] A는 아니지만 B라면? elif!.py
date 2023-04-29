# 변수 answer에 수 1~50 중 하나를 넣어봅시다.
answer = 2

# input을 통해 숫자형으로 입력을 받아서 변수 submit에 저장해봅시다.
submit = input()
submit = int(submit)
# if-elif-else문으로 Up-Down Game을 구현해봅시다.
# 만약 answer보다 submit이 더 크면 "정답보다 더 큰 수를 입력했습니다."
# 만약 answer보다 submit이 더 작으면 "정답보다 더 작은 수를 입력했습니다."
# 만약 answer와 submit이 같으면 "정답!" 를 출력합니다.
if answer < submit :
    print("정답보다 더 큰 수를 입력했습니다.")

elif answer > submit :
    print("정답보다 더 작은 수를 입력했습니다.")

elif answer == submit :
    print("정답!")

else : 
    print("제대로 된 값을 입력하세요.")


