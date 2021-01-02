from snake_env import my_snake

sn1 = my_snake(10)
a = 11
sn1.print_state()
while a!=10:
    a = int(input('Your move:\n'))
    if a not in [0,1,2,10]:
        print('Not Valid a!')
    else:
        life,terminal,reward = sn1.move_step(a)
        sn1.print_state()
        print(sn1.state_flat())
        print('Life: {}, Terminal: {}, Reward: {}\n'.format(life,terminal,reward))
