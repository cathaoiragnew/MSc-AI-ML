package pacman.controllers.examples;

import java.awt.*;
import java.util.ArrayList;

import pacman.controllers.Controller;
import pacman.game.Game;
import pacman.game.GameView;


import static pacman.game.Constants.*;

/* Group details
 * Name: David Cherry, 		ID: 13056212
 * Name: Cathaoir Agnew,	ID: 16171659
 * Name: Conor O'Donovan,	ID: 9730761 
 */


public class CustomPacMan extends Controller<MOVE>
{
    private static final int MIN_DISTANCE = 13;    //if a ghost is this close, run away
    private static final int Large_Radius = 25;
    //private static final int Small_Radius = 15; // this was used for ambush code
    private static final int Hunt_Radius = 10;
    private static final boolean PRINTOUT = true; // set this to false to suppress console printouts during game play

    public MOVE getMove(Game game, long timeDue) {
        int current = game.getPacmanCurrentNodeIndex();

        //get array of power pills for target purposes
        ArrayList<Integer> ptargets = new ArrayList<Integer>();
        int[] powerPills = game.getPowerPillIndices();
        for (int i = 0; i < powerPills.length; i++)            //check with power pills are available
            if (game.isPowerPillStillAvailable(i))
                ptargets.add(powerPills[i]);
        int[] ptargetsArray = new int[ptargets.size()];        //convert from ArrayList to array
        for (int i = 0; i < ptargetsArray.length; i++)
            ptargetsArray[i] = ptargets.get(i);

        
        //Draw Range Lines color coordinated based on range to each ghost
        for (GHOST ghostType : GHOST.values())
            if (game.getGhostLairTime(ghostType) == 0)
                if (game.getShortestPathDistance(current,
                        game.getGhostCurrentNodeIndex(ghostType)) < MIN_DISTANCE)
                    GameView.addLines(game, Color.RED, game.getPacmanCurrentNodeIndex(),
                            game.getGhostCurrentNodeIndex(ghostType));
                else if (game.getShortestPathDistance(current,
                        game.getGhostCurrentNodeIndex(ghostType)) < Large_Radius)
                    GameView.addLines(game, Color.ORANGE, game.getPacmanCurrentNodeIndex(),
                            game.getGhostCurrentNodeIndex(ghostType));
                else
                    GameView.addLines(game, Color.GREEN, game.getPacmanCurrentNodeIndex(),
                            game.getGhostCurrentNodeIndex(ghostType));

        
        //Flee Ghost Code: if any non-edible ghost is too close (less than MIN_DISTANCE), run away
        for (GHOST ghost : GHOST.values())
            if (game.getGhostEdibleTime(ghost) == 0 && game.getGhostLairTime(ghost) == 0)
                if (game.getShortestPathDistance(current, game.getGhostCurrentNodeIndex(ghost)) < MIN_DISTANCE) {
                    if (PRINTOUT) {
                    	System.out.println("Pacman Fleeing"); 
                    }
                    return game.getNextMoveAwayFromTarget(game.getPacmanCurrentNodeIndex(), game.getGhostCurrentNodeIndex(ghost), DM.EUCLID);
                }
        
        
        //Ambush Code: If near a power pill (Min_distance) wait till ghost is near(small radius) before going for it
//		for(GHOST ghost : GHOST.values())
//			if(game.getGhostEdibleTime(ghost)==0 && game.getGhostLairTime(ghost)==0 && game.getNumberOfActivePowerPills()!=0)
//				if(game.getShortestPathDistance(current, game.getClosestNodeIndexFromNodeIndex(current,ptargetsArray,DM.PATH))<Small_Radius)
//					if(game.getShortestPathDistance(current,game.getGhostCurrentNodeIndex(ghost))>Large_Radius)
//					{
//						System.out.println("Move Away From PowerPill");
//						return game.getNextMoveAwayFromTarget(current,game.getClosestNodeIndexFromNodeIndex(current,ptargetsArray,DM.PATH),DM.PATH);
//					}

      
        //Nervous Code: if non-edible ghost is relatively close (Large Radius), run toward power pill
        for (GHOST ghost : GHOST.values())
            if (game.getGhostEdibleTime(ghost) == 0 && game.getGhostLairTime(ghost) == 0 && game.getNumberOfActivePowerPills() != 0)
                if (game.getShortestPathDistance(current, game.getGhostCurrentNodeIndex(ghost)) < Large_Radius) {
                	if (PRINTOUT) {
                		System.out.println("Ghost nearby, moving toward PowerPill"); 
                	}
                    return game.getNextMoveTowardsTarget(current, game.getClosestNodeIndexFromNodeIndex(current, ptargetsArray, DM.PATH), DM.PATH);
                }

       
        //Hunt Code: find the nearest edible ghost and go after them
        int minDistance = Hunt_Radius;//Integer.MAX_VALUE;
        GHOST minGhost = null;
        for (GHOST ghost : GHOST.values())
            if (game.getGhostEdibleTime(ghost) > 0) {
                int distance = game.getShortestPathDistance(current, game.getGhostCurrentNodeIndex(ghost));
                if (distance < minDistance) {
                    minDistance = distance;
                    minGhost = ghost;
                }
            }
        if (minGhost != null) {    //we found an edible ghost
        	if (PRINTOUT) {
        		System.out.println("Hunting Edible Ghost");
        	}
            return game.getNextMoveTowardsTarget(game.getPacmanCurrentNodeIndex(), game.getGhostCurrentNodeIndex(minGhost), DM.PATH);
        }

        
        //Find Pills Code: go after the pills
        ArrayList<Integer> targets = new ArrayList<Integer>();
        int[] pills = game.getPillIndices();
        for (int i = 0; i < pills.length; i++)                    //check which pills are available
            if (game.isPillStillAvailable(i))
                targets.add(pills[i]);
        int[] targetsArray = new int[targets.size()];        //convert from ArrayList to array
        for (int i = 0; i < targetsArray.length; i++)
            targetsArray[i] = targets.get(i);
        if (PRINTOUT) {
        	System.out.println("Finding regular pills");
        }
        return game.getNextMoveTowardsTarget(current, game.getClosestNodeIndexFromNodeIndex(current, targetsArray, DM.PATH), DM.PATH);
    }
}