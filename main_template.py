#
# Podstawy Sztucznej Inteligencji, IIS 2020
# Autor: Tomasz Jaworski
# Opis: Szablon kodu do stabilizacji odwróconego wahadła (patyka) w pozycji pionowej podczas ruchu wózka.
#

import gym # Instalacja: https://github.com/openai/gym
import time
from helper import HumanControl, Keys, CartForce
import matplotlib.pyplot as plt


import numpy as np
import skfuzzy as fuzz

#
# przygotowanie środowiska
#
control = HumanControl()
env = gym.make('gym_PSI:CartPole-v2')
env.reset()
env.render()


def on_key_press(key: int, mod: int):
    global control
    force = 10
    if key == Keys.LEFT:
        control.UserForce = force * CartForce.UNIT_LEFT # krok w lewo
    if key == Keys.RIGHT:
        control.UserForce = force * CartForce.UNIT_RIGHT # krok w prawo
    if key == Keys.P: # pauza
        control.WantPause = True
    if key == Keys.R: # restart
        control.WantReset = True
    if key == Keys.ESCAPE or key == Keys.Q: # wyjście
        control.WantExit = True

env.unwrapped.viewer.window.on_key_press = on_key_press

#########################################################
# KOD INICJUJĄCY - do wypełnienia
#########################################################


# pole angle 

pole_angle_centre = 0;
pole_angle_changer = 5;

pole_angle_range = np.arange(-180.0, 180.1, 0.1)

pole_angle_negative_range = pole_angle_centre - pole_angle_changer
pole_angle_positive_range = pole_angle_centre + pole_angle_changer

pole_angle_negative     = fuzz.trapmf(pole_angle_range, [-180.0, -180.0, pole_angle_negative_range, pole_angle_centre])
pole_angle_zero  = fuzz.trimf(pole_angle_range, [pole_angle_negative_range, pole_angle_centre, pole_angle_positive_range])
pole_angle_positive    = fuzz.trapmf(pole_angle_range, [pole_angle_centre, pole_angle_positive_range, 180.0, 180.0])

# cart_position

cart_position_centre = 0
cart_position_changer = 0.55

cart_position_range = np.arange(-3, 3, 0.01)

cart_position_negative_range = cart_position_centre - cart_position_changer
cart_position_positive_range = cart_position_centre + cart_position_changer

cart_position_negative    = fuzz.trapmf(cart_position_range, [-3, -3, cart_position_negative_range, cart_position_centre])
cart_position_zero  = fuzz.trimf(cart_position_range, [cart_position_negative_range, cart_position_centre, cart_position_positive_range])
cart_position_positive   = fuzz.trapmf(cart_position_range, [cart_position_centre, cart_position_positive_range, 3, 3])

#
# cart_velocity
#
cart_velocity_centre = 0
cart_velocity_changer = 0.5

cart_velocity_range = np.arange(-2, 2, 0.01)

cart_velocity_negative_range = cart_velocity_centre - cart_velocity_changer
cart_velocity_positive_range = cart_velocity_centre + cart_velocity_changer

cart_velocity_negative  = fuzz.trapmf(cart_velocity_range, [-2, -2, cart_velocity_negative_range, cart_velocity_centre])
cart_velocity_zero   = fuzz.trimf(cart_velocity_range, [cart_velocity_negative_range, cart_velocity_centre, cart_velocity_positive_range])
cart_velocity_positive = fuzz.trapmf(cart_velocity_range, [cart_velocity_centre, cart_velocity_positive_range, 2, 2])

#
# force
#

force_centre = 0
force_changer = 5
force_little_changer = 2.5

force_range = np.arange(start=-10.0, stop=10.01, step=0.01)

force_negative_range = 0 - force_changer
force_positive_range = 0 + force_changer
force_little_negative_range = 0 - force_little_changer
force_little_positive_range = 0 + force_little_changer

force_little_negative = fuzz.trapmf(force_range,[-10.0, -10.0, force_little_negative_range, 0])
force_negative = fuzz.trapmf(force_range,[-10.0, -10.0, force_negative_range, 0])
force_zero = fuzz.trimf(force_range, [force_negative_range, 0, force_positive_range])
force_positive = fuzz.trapmf(force_range, [ 0, force_positive_range, 10.0, 10.0])
force_little_positive = fuzz.trapmf(force_range, [ 0, force_little_positive_range, 10.0, 10.0])


"""
1. Określ dziedzinę dla każdej zmiennej lingwistycznej. Każda zmienna ma własną dziedzinę.
2. Zdefiniuj funkcje przynależności dla wybranych przez siebie zmiennych lingwistycznych.
3. Wyświetl je, w celach diagnostycznych.
Przykład wyświetlania:
"""
if True:
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, figsize=(8, 9))

    ax0.plot(pole_angle_range, pole_angle_negative, 'b', linewidth=1.5, label='Negative')
    ax0.plot(pole_angle_range, pole_angle_zero, 'k', linewidth=1.5, label='Zero')
    ax0.plot(pole_angle_range, pole_angle_positive, 'r', linewidth=1.5, label='Positive')
    ax0.set_title('Pole angle')
    ax0.legend()

    ax1.plot(cart_position_range, cart_position_negative, 'b', linewidth=1.5, label='Negative')
    ax1.plot(cart_position_range, cart_position_zero, 'k', linewidth=1.5, label='Zero')
    ax1.plot(cart_position_range, cart_position_positive, 'r', linewidth=1.5, label='Positive')
    ax1.set_title('Cart position')
    ax1.legend()
    
    ax2.plot(cart_velocity_range, cart_velocity_negative, 'b', linewidth=1.5, label='Negative')
    ax2.plot(cart_velocity_range, cart_velocity_zero, 'k', linewidth=1.5, label='Zero')
    ax2.plot(cart_velocity_range, cart_velocity_positive, 'r', linewidth=1.5, label='Positive')
    ax2.set_title('Cart velocity')
    ax2.legend()

    ax3.plot(force_range, force_little_negative, 'g', linewidth=1.5, label='Little Negative')
    ax3.plot(force_range, force_negative, 'b', linewidth=1.5, label='Negative')
    ax3.plot(force_range, force_zero, 'k', linewidth=1.5, label='Zero')
    ax3.plot(force_range, force_positive, 'r', linewidth=1.5, label='Positive')
    ax3.plot(force_range, force_little_positive, 'y', linewidth=1.5, label='Little Positive')
    ax3.set_title('Force range')
    ax3.legend()

    plt.tight_layout()
    plt.show()

    
"""
"""
#########################################################
# KONIEC KODU INICJUJĄCEGO
#########################################################



#
# Główna pętla symulacji
#
while not control.WantExit:

    #
    # Wstrzymywanie symulacji:
    # Pierwsze wciśnięcie klawisza 'p' wstrzymuje; drugie wciśnięcie 'p' wznawia symulację.
    #
    if control.WantPause:
        control.WantPause = False
        while not control.WantPause:
            time.sleep(0.1)
            env.render()
        control.WantPause = False

    #
    # Czy użytkownik chce zresetować symulację?
    if control.WantReset:
        control.WantReset = False
        env.reset()


    ###################################################
    # ALGORYTM REGULACJI - do wypełnienia
    ##################################################

    """
    Opis wektora stanu (env.state)
        cart_position   -   Położenie wózka w osi X. Zakres: -2.5 do 2.5. Ppowyżej tych granic wózka znika z pola widzenia.
        cart_velocity   -   Prędkość wózka. Zakres +- Inf, jednak wartości powyżej +-2.0 generują zbyt szybki ruch.
        pole_angle      -   Pozycja kątowa patyka, a<0 to odchylenie w lewo, a>0 odchylenie w prawo. Pozycja kątowa ma
                            charakter bezwzględny - do pozycji wliczane są obroty patyka.
                            Ze względów intuicyjnych zaleca się konwersję na stopnie (+-180).
        tip_velocity    -   Prędkość wierzchołka patyka. Zakres +- Inf. a<0 to ruch przeciwny do wskazówek zegara,
                            podczas gdy a>0 to ruch zgodny z ruchem wskazówek zegara.
                            
    Opis zadajnika akcji (fuzzy_response):
        Jest to wartość siły przykładana w każdej chwili czasowej symulacji, wyrażona w Newtonach.
        Zakładany krok czasowy symulacji to env.tau (20 ms).
        Przyłożenie i utrzymanie stałej siły do wózka spowoduje, że ten będzie przyspieszał do nieskończoności,
        ruchem jednostajnym.
    """

    cart_position, cart_velocity, pole_angle, tip_velocity = env.state # Wartości zmierzone

    if cart_position > 3 or cart_position < -3:
        env.reset()
    if pole_angle > 3 or pole_angle < -3:
        env.reset()

    pole_angle = pole_angle * 180.0 / np.math.pi;

    """
    
    1. Przeprowadź etap rozmywania, w którym dla wartości zmierzonych wyznaczone zostaną ich przynależności do poszczególnych
       zmiennych lingwistycznych. Jedno fizyczne wejście (źródło wartości zmierzonych, np. położenie wózka) posiada własną
       zmienną lingwistyczną.
       
       Sprawdź funkcję interp_membership
       
    2. Wyznacza wartości aktywacji reguł rozmytych, wyznaczając stopień ich prawdziwości.
       Przykład reguły:
       JEŻELI kąt patyka jest zerowy ORAZ prędkość wózka jest zerowa TO moc chwilowa jest zerowa
       JEŻELI kąt patyka jest lekko ujemny ORAZ prędkość wózka jest zerowa TO moc chwilowa jest lekko ujemna
       JEŻELI kąt patyka jest średnio ujemny ORAZ prędkość wózka jest lekko ujemna TO moc chwilowa jest średnio ujemna
       JEŻELI kąt patyka jest szybko rosnący w kierunku ujemnym TO moc chwilowa jest mocno ujemna
       .....
       
       Przyjmując, że spójnik LUB (suma rozmyta) to max() a ORAZ/I (iloczyn rozmyty) to min() sprawdź funkcje fmax i fmin.
    
    
    3. Przeprowadź agregację reguł o tej samej konkluzji.
       Jeżeli masz kilka reguł, posiadających tę samą konkluzję (ale różne przesłanki) to poziom aktywacji tych reguł
       należy agregować tak, aby jedna konkluzja miała jeden poziom aktywacji. Skorzystaj z sumy rozmytej.
    
    4. Dla każdej reguły przeprowadź operację wnioskowania Mamdaniego.
       Operatorem wnioskowania jest min().
       Przykład: Jeżeli lingwistyczna zmienna wyjściowa ForceToApply ma 5 wartości (strong left, light left, idle, light right, strong right)
       to liczba wyrażeń wnioskujących wyniesie 5 - po jednym wywołaniu operatora Mamdaniego dla konkluzji.
       
       W ten sposób wyznaczasz aktywacje poszczególnych wartości lingwistycznej zmiennej wyjściowej.
       Uważaj - aktywacja wartości zmiennej lingwistycznej w konkluzji to nie liczba a zbiór rozmyty.
       Ponieważ stosujesz operator min(), to wynikiem będzie "przycięty od góry" zbiór rozmyty. 
       
    7. Czym będzie wyjściowa wartość skalarna?
    
    """

     # pole angle
    is_pole_angle_left =     fuzz.interp_membership(pole_angle_range, pole_angle_negative,           pole_angle)
    is_pole_angle_vertical = fuzz.interp_membership(pole_angle_range, pole_angle_zero,       pole_angle)
    is_pole_angle_right =    fuzz.interp_membership(pole_angle_range, pole_angle_positive,          pole_angle)
    print(
        f"pole angle [{is_pole_angle_left:8.4f} {is_pole_angle_vertical:8.4f} {is_pole_angle_right:8.4f}]")

    # cart position
    is_cart_position_left =    fuzz.interp_membership(cart_position_range, cart_position_negative,         cart_position)
    is_cart_position_desired = fuzz.interp_membership(cart_position_range, cart_position_zero,      cart_position)
    is_cart_position_right =   fuzz.interp_membership(cart_position_range, cart_position_positive,        cart_position)
    print(
        f"cart posit [{is_cart_position_left:8.4f} {is_cart_position_desired:8.4f} {is_cart_position_right:8.4f}]")

    # cart velocity
    is_cart_velocity_left =    fuzz.interp_membership(cart_velocity_range, cart_velocity_negative,         cart_velocity)
    is_cart_velocity_zero =    fuzz.interp_membership(cart_velocity_range, cart_velocity_zero,         cart_velocity)
    is_cart_velocity_right =   fuzz.interp_membership(cart_velocity_range, cart_velocity_positive,        cart_velocity)
    print(
        f"cart veloc [{is_cart_velocity_left:8.4f} {is_cart_velocity_zero:8.4f} {is_cart_velocity_right:8.4f}]")

    # 1. JEŻELI kąt patyka jest ujemny TO siła jest ujemna
    # 2. JEŻELI kąt patyka jest zerowy ORAZ pozycja wózka jest ujemna TO siła jest lekko ujemna
    # 3. JEŻELI kąt patyka jest dodatni TO siła jest dodatnia
    # 4. JEŻELI kąt patyka jest zerowy ORAZ pozycja wózka jest dodatnia TO siła jest lekko dodatnia

    r1 = max([is_pole_angle_left]);
    r2 = min([is_pole_angle_vertical, is_cart_position_left]);
    r3 = max([is_pole_angle_right]);
    r4 = min([is_pole_angle_vertical, is_cart_position_right]);

    little_negative = max([r2]);
    negative = max([r1]);
    zero = max([0]);
    positive = max([r3]);
    little_positive = max([r4]);

    u_force_little_negative = np.fmin(force_little_negative, little_negative);
    u_force_negative = np.fmin(force_negative, negative);
    u_force_zero = np.fmin(force_zero, zero);
    u_force_positive = np.fmin(force_positive, positive);
    u_force_little_positive = np.fmin(force_little_positive, little_positive);


#   5. Agreguj wszystkie aktywacje dla danej zmiennej wyjściowej.
    result = np.maximum.reduce(
        [u_force_little_negative, u_force_negative, u_force_zero, u_force_positive, u_force_little_positive])
    
#   6. Dokonaj defuzyfikacji (np. całkowanie ważone - centroid).
    fuzzy_response = fuzz.centroid(force_range, result)

#    fuzzy_response = CartForce.IDLE_FORCE # do zmiennej fuzzy_response zapisz wartość siły, jaką chcesz przyłożyć do wózka.

    #
    # KONIEC algorytmu regulacji
    #########################

    # Jeżeli użytkownik chce przesunąć wózek, to jego polecenie ma wyższy priorytet
    if control.UserForce is not None:
        applied_force = control.UserForce
        control.UserForce = None
    else:
        applied_force = fuzzy_response

    #
    # Wyświetl stan środowiska oraz wartość odpowiedzi regulatora na ten stan.
    print(
        f"cpos={cart_position:8.4f}, cvel={cart_velocity:8.4f}, pang={pole_angle:8.4f}, tvel={tip_velocity:8.4f}, force={applied_force:8.4f}")

    #
    # Wykonaj krok symulacji
    env.step(applied_force)

    #
    # Pokaż kotku co masz w środku
    env.render()

#
# Zostaw ten patyk!
env.close()

