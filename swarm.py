#! /usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import division

try:
    import pygame as pg
    import numpy as np
except ImportError:
    raise ImportError("PyGame and NumPy are required")

import sys, math, random, operator


DEBUG = True
N_BEES = 100

GRID_WIDTH = 340
GRID_HEIGHT = 240
VIEW_WIDTH = 300
VIEW_HEIGHT = 200
VIEW_MARGIN = 20

COLOR_BEE_IDLE = 0x854500
COLOR_BEE_YOUNG = 0xb58010
COLOR_BEE_HUNGRY = 0xff0000
COLOR_GOAL = 0x00ffff
WHITE = 0xffffff
BLACK = 0x000000


################################################################################
############################ HELPER FUNCTIONS ##################################
################################################################################

dIdx_to_translation = [(-1,-1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]

def radToIdx(r):
    """
    Return a standard index number ([0-7] starting at North-West direction)
    from an angle in radians
    
    Args:
        r: angle in radians
    """
    return int(round(3 + 4*r/math.pi))%8

def randomAdn(clan=1):
    return [(clan, clan)] + \
        [(random.randint(0,255), random.randint(0,255)) for i in range(8)]

def goodAdn():
    """Proliferating type Adn"""
    return [(1, 1),
            (83, 239), (4, 127),
            (179, 206), (233, 135),
            (132, 14), (173, 167),
            (118, 53), (189, 38)]

def recombination(adn1, adn2):
    """Recombinates two adn sets to form a new adn"""
    new_adn = list()
    for (w1,b1), (w2,b2) in zip(adn1, adn2):
        mask1 = random.randrange(256)
        mask2 = random.randrange(256)
        wr = (w1&mask1) | (w2&~mask1)
        br = (b1&mask2) | (b2&~mask2)
        new_adn.append( (wr, br) )
    return new_adn

def blur(a):
    """Blur a standard indices array.
       This function has terrible performance when used with
       numpy's float types so it's best to convert the argument
       to a python list with the tolist() method first.
    """
    r = [(a[7]+a[0]+a[1])/3.,
         (a[0]+a[1]+a[2])/3.,
         (a[1]+a[2]+a[3])/3.,
         (a[2]+a[3]+a[4])/3.,
         (a[3]+a[4]+a[5])/3.,
         (a[4]+a[5]+a[6])/3.,
         (a[5]+a[6]+a[7])/3.,
         (a[6]+a[7]+a[0])/3.]
    return np.array(r)

def dist(x1, y1, x2, y2):
    """Calculate the euclidiean distance between two points."""
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)
    
def dir(xo, yo, xd, yd):
    return radToIdx(math.atan2(yd-yo, xd-xo))

def randomArray():
    return np.random.random_sample(8)

def clamp(val, lower, upper):
    """Clamp a given value between lower and upper boundaries."""
    return min(upper, max(lower, val))


################################################################################
############################### CLASS BEE ######################################
################################################################################

# ADN loci:
#   [0]	Clan
#   [1]	Foraging
#   [2]	Aggressivity
#   [3]	Horniness
#   [4]	Leader attraction
#   [5]	Bees attraction
#   [6]	Stress sensitivity
#   [7]	Feeding/Healing
#   [8] Explore
(ADN_CLAN, ADN_FORA, ADN_AGGR, ADN_MATE, ADN_LATTR,
    ADN_BATTR, ADN_STRESS, ADN_FEED, ADN_EXPL) = range(9)


class Bee:
    ENERGY_INIT = 800
    ENERGY_MAX = 4000
    STRESS_MAX = 400
    AGE_ADULT = 500
    AGE_MAX =  10000
    CURSOR_CRITICAL_DIST = 400
    
    def __init__(self, x, y, adn, world, generation=0):
        self.world = world
        self.alive = True
        self.generation = generation
        self.x = x
        self.y = y
        self.dIdx = random.randint(0, 7)	# direction index 
        self.age = 0
        self.energy = self.ENERGY_INIT
        self.stress = 0
        self.food_memory = []
        self.feed_lock = False			# prevent back and forth feeding 
        self.color = COLOR_BEE_YOUNG
        
        self.adn = adn
    
    def __repr__(self):
        return "<Bee>"
    
    def adnW(self, locus, signed=False):
        """Returns gene's weight."""
        if signed:
            return self.adn[locus][0] - 127
        return self.adn[locus][0]
    
    def adnB(self, locus):
        """Returns gene's bias (can be negative)."""
        return self.adn[locus][1] - 127
    
    def getFoodScore(self):
        # XXX: UNUSED
        r = np.ones(8, dtype=np.float32)
        dists = np.array([dist(x, y, self.x, self.y) for x, y in self.food_memory])
        i = len(dists)
        for idx in dists.argsort():
            r[idx] += i
            i -= 1
        return r/(len(dists)+1)
    
    def update(self):
        ## UPDATE INTERNAL STATES
        self.age += 1
        self.energy -= 1
        self.stress -= 1
        if self.energy < 200:
            self.stress += 2
        self.stress = clamp(self.stress, 0, self.STRESS_MAX)
        
        self.isMoving = True # XXX
        
        surrounding = self.world.getAround(self.x, self.y)
        
        ## Bee arrived at destination
        if surrounding.atGoal:
            self.world.saveBee(self)
            self.alive = False
            return
        
        ## is it alive ?
        if self.age>self.AGE_MAX or self.energy<=0 or surrounding.free==0:
            self.alive = False
            return
        
        ## WORLD AROUND ME
        cursor_dist = dist(self.world.mX, self.x, self.world.mY, self.y)
        cursor_dir = radToIdx(math.atan2(self.world.mY-self.y, self.world.mX-self.x))
        
        ## UPDATE DRIVES
        drives = dict()
        #### FORAGE
        drives['eat'] = \
            self.adnB(ADN_FORA) + self.adnW(ADN_FORA)*(1-self.energy/self.ENERGY_MAX)
        
        if len(surrounding.neighbours) > 0:
            #### MATE
            if self.age > self.AGE_ADULT and \
               self.energy > self.ENERGY_INIT and \
               surrounding.free > 4:
                drives['mate'] = \
                    self.adnB(ADN_MATE) + \
                    self.adnW(ADN_MATE) * \
                    ((self.energy/self.ENERGY_MAX)-(self.stress/self.STRESS_MAX))
            #### FEED
            drives['feed'] = \
                self.adnB(ADN_FEED) + \
                self.adnW(ADN_FEED) * \
                ((self.energy/self.ENERGY_MAX)-(self.stress/self.STRESS_MAX))
            #### ATTACK
        
        #### EXPLORE
        drives['explore'] = \
            self.adnB(ADN_EXPL) + \
            self.adnW(ADN_EXPL) * \
            ((self.energy/self.ENERGY_MAX)-(self.stress/self.STRESS_MAX))
        #### FOLLOW LEADER
        drives['f_leader'] = (
            self.adnB(ADN_LATTR) +
            self.adnW(ADN_LATTR) *
            ((self.stress/self.STRESS_MAX) + min(1, cursor_dist/self.CURSOR_CRITICAL_DIST)))
        #### FOLLOW NEIGHBOURS
        drives['f_neigh'] = self.adnB(ADN_BATTR) + \
            self.adnW(ADN_BATTR,True)*(self.stress/self.STRESS_MAX)
        
        ## EXECUTE ACTION
        action = sorted(drives.items(), key=operator.itemgetter(1), reverse=True)[0][0]
        if action == 'eat':
            # Check if there's food around
            if len(surrounding.food)>0 and self.energy<self.ENERGY_MAX:
                self.energy += random.choice(surrounding.food).take(50)
                self.isMoving = False
        elif action == 'mate':
            # Choose a neighbour partner bee from
            partner = random.choice(surrounding.neighbours)[1]
            # Look for the 'clearest' position around
            idx = blur(surrounding.obstacles.tolist()).argmin()
            spawn_x = self.x+dIdx_to_translation[idx][0]
            spawn_y = self.y+dIdx_to_translation[idx][1]
            new_adn = recombination(self.adn, partner.adn)
            child_bee = Bee(spawn_x, spawn_y, new_adn,
                            self.world, generation=self.generation+1)
            self.world.addBee(child_bee)
            self.energy -= self.ENERGY_INIT
            self.isMoving = False
        elif action == 'feed' and not self.feed_lock:
            partner = sorted(zip(*surrounding.neighbours)[1],
                             key=operator.attrgetter('energy'))[0]
            if self.energy > partner.energy+50:
                self.energy -= 50
                partner.energy += 50
                partner.stress -= 2
                partner.feed_lock = True
                self.isMoving = False

        self.feed_lock = False
        
        if self.isMoving:
            sum_score = np.zeros(8)
            # Follow leader (ignored if drive is negative)
            # The bigger the angle between an idx and the leader direction
            # the lower the score will be for that idx
            if drives['f_leader'] > 0:
                l_score = np.arange(8, dtype=np.float32) - cursor_dir
                l_score %= 8
                l_score -= 4
                l_score = abs(l_score)/4.	# scores range from 0.0 (180°) to 1.0 (0°)
                sum_score += l_score * drives['f_leader']
            
            # Proximity with other bees, scores range from 0.0 (furthest) to 1.0 (nearest)
            if len(surrounding.neighbours) > 0:
                # Here we use a python list rather than a ndarray for better
                # performance with the blur function
                p_score = [0]*8
                for i,_ in surrounding.neighbours:
                    p_score[i] = 1.0
                p_score = blur(p_score)
                sum_score += p_score * drives['f_neigh']
             
            # Random walk
            sum_score += randomArray() * drives['explore']
            
            # Move toward food
            #sum_score += self.getFoodScore() * drives['eat']
            
            # find best direction
            ## argsort: return array of indices that would sort the array
            for idx in sum_score.argsort()[::-1]:
                if not surrounding.obstacles[idx]:  # maybe we don't need that
                    self.dIdx = idx
                    self.x += dIdx_to_translation[self.dIdx][0]
                    self.y += dIdx_to_translation[self.dIdx][1]
                    break
        
        # color setting
        if self.energy < 500 and (self.age%4) == 0:
            self.color = COLOR_BEE_HUNGRY
        elif self.age<self.AGE_ADULT:
            self.color = COLOR_BEE_YOUNG
        else:
            self.color = COLOR_BEE_IDLE
        

################################################################################
################################ CLASS SURROUND ################################
################################################################################

class Surround:
    def __init__(self, entities, obstacles):
        """
        Args:
            entities: list of surrounding entities (idx, object)
            obstacles: binary array of obstacles (1 if obstacle, 0 if not)
        """
        self.neighbours = list()
        self.food = list()
        self.obstacles = obstacles
        self.free = 8 - len(obstacles.nonzero()[0])
        self.atGoal = False
        for e in entities:
            if isinstance(e[1], Bee):
                self.neighbours.append(e)
            elif isinstance(e[1], Food):
                self.food.append(e[1])
            elif e[1] == COLOR_GOAL:
                self.atGoal = True


################################################################################
################################ CLASS FOOD ####################################
################################################################################        

class Food:
    def __init__(self, x, y, value=1000):
        self.x = x
        self.y = y
        self.value = value
        self.color = 0xffff00
        self.eaten = False
    
    def take(self, q):
        if q>self.value:
            q = self.value
        self.value -= q
        if not self.eaten:
            self.eaten = True
            self.color = 0xeecc00
        return q


################################################################################
################################ CLASS WORLD ###################################
################################################################################        
    
class World:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.mX = 0	# mouse cursor
        self.mY = 0	# mouse cursor
        self.entities = np.zeros(shape=(height, width), dtype=object)
        self.rendered = np.empty(shape=(height, width), dtype=np.int32)
        self.rendered.fill(WHITE)
        self.saved_bees = list() # list of bees that reached the goal
        self.bees = list()	# list of living bees in this world
        self.food = list()	# list of food in this world
    
    def setCursor(self, x, y):
        self.mX = x
        self.mY = y
    
    def isInWindow(self, x, y, xmin=0, ymin=0, width=0, height=0):
        """Check if the given position is included in the given window."""
        if not width: width = self.width
        if not height: height = self.height
        return x>=xmin and x<(xmin+width) and y>=ymin and y<(ymin+height)
    
    def newBee(self, x, y):
        self.addBee(Bee(x, y, randomAdn(), self))
    
    def saveBee(self, bee):
        self.saved_bees.append(bee)
    
    def addBee(self, bee):
        self.bees.append(bee)
        self.entities[bee.y, bee.x] = bee
        bee.alive = True
    
    def addFood(self, x, y):
        food = Food(x, y)
        self.food.append(food)
        self.entities[y, x] = food
        self.rendered[y, x] = food.color
    
    def addObstacle(self, x, y):
        self.rendered[y, x] = BLACK
    
    def addGoal(self, x, y):
        self.entities[y, x] = COLOR_GOAL
        self.rendered[y, x] = COLOR_GOAL
    
    def getAround(self, x, y):
        """Returns a Surround object given a position."""
        ents = list()
        obs =  np.zeros(shape=8)
        for i, (dx, dy) in enumerate(dIdx_to_translation):
            if not self.isInWindow(x+dx, y+dy):
                obs[i] = 1
                continue
            if self.rendered[y+dy, x+dx] != WHITE:
                obs[i] = 1
            if self.entities[y+dy, x+dx]:
                ents.append( (i, self.entities[y+dy, x+dx]) )
        return Surround(ents, obs)
    
    def renderAt(self, surface, x, y):
        """
        Render the world grid on the given surface.
           
        Args:
           surface: pygame surface to be rendered
           x,y: view window position (in world grid coordinates)
        """
        w = surface.get_width()
        h = surface.get_height()
        
        window = np.zeros(shape=(h,w), dtype=np.int32)
        window[-min(y,0):self.height-y, -min(x,0):self.width-x] = \
            self.rendered[max(0,y):min(self.height, h+y), max(0,x):min(self.width, w+x)]
        
        pg.surfarray.blit_array(surface, window.transpose())
    
    def tick(self):
        for i, b in enumerate(self.bees):
            self.entities[b.y, b.x] = 0
            self.rendered[b.y, b.x] = WHITE
            b.update()
            if b.alive:
                self.entities[b.y, b.x] = b
                self.rendered[b.y, b.x] = b.color
            else:
                del self.bees[i]
        for i, f in enumerate(self.food):
            if f.value <= 0:
                self.entities[f.y, f.x] = 0
                self.rendered[f.y, f.x] = WHITE
                del self.food[i]
            elif f.eaten:
                self.rendered[f.y, f.x] = f.color
            


################################################################################
############################# CLASS GAME ####################################### 
################################################################################

class Game:
    def __init__(self, surf, level=None):
        if level:
            spawn_loc, food_loc, obstacle_loc, goal_loc = self.loadLevel(level)
            self.width = len(level[0])
            self.height = len(level)
            self.world = World(self.width, self.height)
            for x, y in obstacle_loc:
                self.world.addObstacle(x, y)
            for x, y in food_loc:
                self.world.addFood(x, y)
            for x, y in spawn_loc:
                self.world.newBee(x, y)
            for x, y in goal_loc:
                self.world.addGoal(x, y)
        else:
            self.width = GRID_WIDTH
            self.height = GRID_HEIGHT
            self.world = World(GRID_WIDTH, GRID_HEIGHT)
            # Populate with default number of bees
            n_bees = N_BEES
            while n_bees>0:
                x = random.randint(0, VIEW_WIDTH-1)
                y = random.randint(0, VIEW_HEIGHT-1)
                # Avoid creating bees at same location than food and other bees
                if not self.world.entities[y, x]:
                    self.world.newBee(x, y)
                    n_bees -= 1
        
        self.x = self.width/2 - VIEW_WIDTH/2
        self.y = self.height/2 - VIEW_HEIGHT/2
        self.screen = surf
        self.screen1 = pg.Surface((VIEW_WIDTH, VIEW_HEIGHT))	# unscaled screen surface
        self.clock = pg.time.Clock()
        #self.font = pg.font.Font(None, 16)
        self.sysfont = pg.font.SysFont("FreeMono, Monospace", 10)
    
    def loadLevel(self, level):
        spawn_loc = []
        food_loc = []
        obstacle_loc = []
        goal_loc = []
        width, height = len(level[0]), len(level)
        for j in range(height):
            for i in range(width):
                data = level[j][i]
                if data == ' ': pass
                elif data == 'o': spawn_loc.append((i, j))
                elif data == 'f': food_loc.append((i, j))
                elif data == 'X': obstacle_loc.append((i, j))
                elif data == 'G': goal_loc.append((i, j))
        return (spawn_loc, food_loc, obstacle_loc, goal_loc)
    
    def saveAdn(self, bees_list):
        str_format = "{:10} {:10} {:10} {:10} {:10} {:10} {:10} {:10} {:10}"
        adn = [str_format.format("clan", "forag", "aggr", "mate",\
                                 "f_lead", "f_neigh", "stress", "feed", "explo")+'\n']
        for b in bees_list:
            str = str_format.format(*b.adn)
            str += "\t[{}]\n".format(b.generation)
            adn.append(str)
        with open("adn.txt", "w") as f:
            f.writelines(adn)
    
    def main(self):
        paused = False
        running = True
        # GAME OVER text surface
        game_over = pg.Surface.convert(self.sysfont.render("GAME OVER", False, (255, 0, 0)))
        go_w, go_h = game_over.get_size()
        # Cool font effect !
        game_over = pg.transform.smoothscale(game_over, (go_w*2, go_h*2))
        game_over = pg.transform.scale(game_over, (go_w*4, go_h*4))
        go_w, go_h = game_over.get_size()
        
        while running:
            self.clock.tick(30)
            fps = round(self.clock.get_fps())
            
            # move view window if mouse is near the borders
            mx, my = pg.mouse.get_pos()
            if mx > 200 + VIEW_WIDTH: self.x += 6
            if mx < VIEW_WIDTH - 200: self.x -= 6
            if my > 110 + VIEW_HEIGHT: self.y += 6
            if my < VIEW_HEIGHT - 110: self.y -= 6
            
            # clamp window coordinates to valid positions
            self.x = clamp(self.x, -VIEW_MARGIN, VIEW_MARGIN+self.width-VIEW_WIDTH)
            self.y = clamp(self.y, -VIEW_MARGIN, VIEW_MARGIN+self.height-VIEW_HEIGHT)
            
            # divide cursor position by 2 to compensate for the scale2x of the screen
            self.world.setCursor(self.x+mx/2, self.y+my/2)
            
            # read keyboard
            if pg.key.get_focused():
                keys = pg.key.get_pressed()
                # Stress Button
                if keys[pg.K_s]:
                    print("Max stress !")
                    for bee in self.world.bees:
                        bee.stress = Bee.STRESS_MAX
                # Pause Button
                if keys[pg.K_p]:
                    paused = not paused
                    pg.time.wait(150)
            
            for e in pg.event.get():
                if e.type == pg.QUIT:
                    self.saveAdn(self.world.bees)
                    pg.display.quit()
                    pg.quit()
                    sys.exit()
            
            if not paused:
                self.world.tick()
            self.world.renderAt(self.screen1, self.x, self.y)
            if len(self.world.bees) > 0:
                # Add the HUD
                hud_bee_count = self.sysfont.render(
                    str(len(self.world.bees)), False, (255,0,0))
                self.screen1.blit(hud_bee_count, (275,185))
                if len(self.world.saved_bees) > 0:
                    hud_saved_bees = self.sysfont.render(
                        str(len(self.world.saved_bees)), False, (255,0,0))
                    self.screen1.blit(hud_saved_bees, (5, 185))
                if DEBUG:
                    hud_fps = self.sysfont.render(str(fps)+" fps", False, (255, 0, 0))
                    self.screen1.blit(hud_fps, (5, 5))
            elif len(self.world.saved_bees) > 0:
                # WIN
                self.saveAdn(self.world.saved_bees)
                running = False
            else:
                # GAME OVER
                t = pg.time.get_ticks() // 1000
                if t%2 == 0:
                    text_x = (VIEW_WIDTH-go_w) // 2
                    text_y = (VIEW_HEIGHT-go_h) // 2
                    self.screen1.blit(game_over, (text_x, text_y))
                paused = True
            pg.transform.scale2x(self.screen1, self.screen)
            #pg.transform.smoothscale(self.screen1, (VIEW_WIDTH*2, VIEW_HEIGHT*2), self.screen)
            #pg.transform.scale(self.screen1, (VIEW_WIDTH*2, VIEW_HEIGHT*2), self.screen)
            pg.display.flip()


################################################################################
################################## MAIN ######################################## 
################################################################################

def main(level=None):
    pg.init()
    screen = pg.display.set_mode((VIEW_WIDTH*2, VIEW_HEIGHT*2))
    pg.display.set_caption("Swarm")
    game = Game(screen, level)
    game.main()


if __name__ == '__main__':
    level = None
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as f:
            level = [l[:-1] for l in f.readlines() if l.endswith('\n')]
    main(level)