//Written by Zhipeng Wang, May 1st, 2015


#include <cilk.h>
#include <reducer_max.h>

extern "C"{

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

}


#define BIT 0x1

#define X_BLACK 0
#define O_WHITE 1
#define OTHERCOLOR(c) (1-(c))

/* 
	represent game board squares as a 64-bit unsigned integer.
	these macros index from a row,column position on the board
	to a position and bit in a game board bitvector
*/
#define BOARD_BIT_INDEX(row,col) ((8 - (row)) * 8 + (8 - (col)))
#define BOARD_BIT(row,col) (0x1LL << BOARD_BIT_INDEX(row,col))
#define MOVE_TO_BOARD_BIT(m) BOARD_BIT(m.row, m.col)

/* all of the bits in the row 8 */
#define ROW8 \
  (BOARD_BIT(8,1) | BOARD_BIT(8,2) | BOARD_BIT(8,3) | BOARD_BIT(8,4) |	\
   BOARD_BIT(8,5) | BOARD_BIT(8,6) | BOARD_BIT(8,7) | BOARD_BIT(8,8))
			  
/* all of the bits in column 8 */
#define COL8 \
  (BOARD_BIT(1,8) | BOARD_BIT(2,8) | BOARD_BIT(3,8) | BOARD_BIT(4,8) |	\
   BOARD_BIT(5,8) | BOARD_BIT(6,8) | BOARD_BIT(7,8) | BOARD_BIT(8,8))

/* all of the bits in column 1 */
#define COL1 (COL8 << 7)

#define IS_MOVE_OFF_BOARD(m) (m.row < 1 || m.row > 8 || m.col < 1 || m.col > 8)
#define IS_DIAGONAL_MOVE(m) (m.row != 0 && m.col != 0)
#define MOVE_OFFSET_TO_BIT_OFFSET(m) (m.row * 8 + m.col)

typedef unsigned long long ull;

/* 
	game board represented as a pair of bit vectors: 
	- one for x_black disks on the board
	- one for o_white disks on the board
*/
typedef struct { ull disks[2]; } Board;

typedef struct { int row; int col; } Move;

Board start = { 
	BOARD_BIT(4,5) | BOARD_BIT(5,4) /* X_BLACK */, 
	BOARD_BIT(4,4) | BOARD_BIT(5,5) /* O_WHITE */
};
 
Move offsets[] = {
  {0,1}		/* right */,		{0,-1}		/* left */, 
  {-1,0}	/* up */,		{1,0}		/* down */, 
  {-1,-1}	/* up-left */,		{-1,1}		/* up-right */, 
  {1,1}		/* down-right */,	{1,-1}		/* down-left */
};

int noffsets = sizeof(offsets)/sizeof(Move);
char diskcolor[] = { '.', 'X', 'O', 'I' };


void PrintDisk(int x_black, int o_white)
{
  printf(" %c", diskcolor[x_black + (o_white << 1)]);
}

void PrintBoardRow(int x_black, int o_white, int disks)
{
  if (disks > 1) {
    PrintBoardRow(x_black >> 1, o_white >> 1, disks - 1);
  }
  PrintDisk(x_black & BIT, o_white & BIT);
}

void PrintBoardRows(ull x_black, ull o_white, int rowsleft)
{
  if (rowsleft > 1) {
    PrintBoardRows(x_black >> 8, o_white >> 8, rowsleft - 1);
  }
  printf("%d", rowsleft);
  PrintBoardRow((int)(x_black & ROW8),  (int) (o_white & ROW8), 8);
  printf("\n");
}

void PrintBoard(Board b)
{
  printf("  1 2 3 4 5 6 7 8\n");
  PrintBoardRows(b.disks[X_BLACK], b.disks[O_WHITE], 8);
}

/* 
	place a disk of color at the position specified by m.row and m,col,
	flipping the opponents disk there (if any) 
*/
void PlaceOrFlip(Move m, Board *b, int color) 
{
  ull bit = MOVE_TO_BOARD_BIT(m);
  b->disks[color] |= bit;
  b->disks[OTHERCOLOR(color)] &= ~bit;
}

/* 
	try to flip disks along a direction specified by a move offset.
	the return code is 0 if no flips were done.
	the return value is 1 + the number of flips otherwise.
*/
int TryFlips(Move m, Move offset, Board *b, int color, int verbose, int domove)
{
  Move next;
  next.row = m.row + offset.row;
  next.col = m.col + offset.col;

  if (!IS_MOVE_OFF_BOARD(next)) {
    ull nextbit = MOVE_TO_BOARD_BIT(next);
    if (nextbit & b->disks[OTHERCOLOR(color)]) {
      int nflips = TryFlips(next, offset, b, color, verbose, domove);
      if (nflips) {
	if (verbose) printf("flipping disk at %d,%d\n", next.row, next.col);
	if (domove) PlaceOrFlip(next,b,color);
	return nflips + 1;
      }
    } else if (nextbit & b->disks[color]) return 1;
  }
  return 0;
} 


// Return the whole number of flips at one move

int FlipDisks(Move m, Board *b, int color, int verbose, int domove)
{
  int i;
  int nflips = 0;
	
  /* try flipping disks along each of the 8 directions */
  for(i=0;i<noffsets;i++) {
    int flipresult = TryFlips(m,offsets[i], b, color, verbose, domove);
    nflips += (flipresult > 0) ? flipresult - 1 : 0;
  }
  return nflips;
}

// ReadMove for computer;

void CReadMove(int color, Board *b, Move *bestm)
{
  int nflips = FlipDisks(*bestm, b,color, 1, 1);
  PlaceOrFlip(*bestm, b, color);
  printf("You flipped %d disks\n", nflips);
  PrintBoard(*b);

}

void ReadMove(int color, Board *b)
{
  Move m;
  ull movebit;
  for(;;) {
    printf("Enter %c's move as 'row,col': ", diskcolor[color+1]);
    scanf("%d,%d",&m.row,&m.col);
		
    /* if move is not on the board, move again */
    if (IS_MOVE_OFF_BOARD(m)) {
      printf("Illegal move: row and column must both be between 1 and 8\n");
      PrintBoard(*b);
      continue;
    }
    movebit = MOVE_TO_BOARD_BIT(m);
		
    /* if board position occupied, move again */
    if (movebit & (b->disks[X_BLACK] | b->disks[O_WHITE])) {
      printf("Illegal move: board position already occupied.\n");
      PrintBoard(*b);
      continue;
    }
		
    /* if no disks have been flipped */ 
    {
      int nflips = FlipDisks(m, b,color, 1, 1);
      if (nflips == 0) {
	printf("Illegal move: no disks flipped\n");
	PrintBoard(*b);
	continue;
      }
      PlaceOrFlip(m, b, color);
      printf("You flipped %d disks\n", nflips);
      PrintBoard(*b);
    }
    break;
  }
}

/*
	return the set of board positions adjacent to an opponent's
	disk that are empty. these represent a candidate set of 
	positions for a move by color.
*/
Board NeighborMoves(Board b, int color)
{
  int i;
  Board neighbors = {0,0};
  for (i = 0;i < noffsets; i++) {
    ull colmask = (offsets[i].col != 0) ? 
      ((offsets[i].col > 0) ? COL1 : COL8) : 0;
    int offset = MOVE_OFFSET_TO_BIT_OFFSET(offsets[i]);

    if (offset > 0) {
      neighbors.disks[color] |= 
	(b.disks[OTHERCOLOR(color)] >> offset) & ~colmask;
    } else {
      neighbors.disks[color] |= 
	(b.disks[OTHERCOLOR(color)] << -offset) & ~colmask;
    }
  }
  neighbors.disks[color] &= ~(b.disks[X_BLACK] | b.disks[O_WHITE]);
  return neighbors;
}

/*
	return the set of board positions that represent legal
	moves for color. this is the set of empty board positions  
	that are adjacent to an opponent's disk where placing a
	disk of color will cause one or more of the opponent's
	disks to be flipped.
*/
int EnumerateLegalMoves(Board b, int color, Board *legal_moves)
{
  static Board no_legal_moves = {0,0};
  Board neighbors = NeighborMoves(b, color);
  ull my_neighbor_moves = neighbors.disks[color];
  int row;
  int col;
	
  int num_moves = 0;
  *legal_moves = no_legal_moves;
	
  for(row=8; row >=1; row--) {
    ull thisrow = my_neighbor_moves & ROW8;
    for(col=8; thisrow && (col >= 1); col--) {
      if (thisrow & COL8) {
	Move m = { row, col };
	if (FlipDisks(m, &b, color, 0, 0) > 0) {
	  legal_moves->disks[color] |= BOARD_BIT(row,col);
	  num_moves++;
	}
      }
      thisrow >>= 1;
    }
    my_neighbor_moves >>= 8;
  }
  return num_moves;
}

// This is for the computer part, each time it will select the move with the maximum flippings;

int CEnumerateLegalMoves(Board b, int color, Move *bestm, int index, int fix_index)
{

  // In the end;
  if (index < 0){

    return 0;

  }
  // In the begining;
  else if(index == fix_index){

    Board neighbors = NeighborMoves(b, color);
    ull my_neighbor_moves = neighbors.disks[color];
    int row;
    int col;
    
    int num_flip = -65;
    int num_moves = 0;
    
    for(row=8; row >=1; row--) {
      ull thisrow = my_neighbor_moves & ROW8;
      for(col=8; thisrow && (col >= 1); col--) {

	if (thisrow & COL8) {
	  Move m = { row, col };


	  // Here is the tricky point, the FlipDisks function doesn't consider the fact that the board position is already occupied, and if it is moved out of board, so we need to consider about that ;

	  /* if board position occupied */
	  ull movebit = MOVE_TO_BOARD_BIT(m);


	  if (movebit & (b.disks[X_BLACK] | b.disks[O_WHITE])) {


	    continue;

	  }

	  /* if move is not on the board */
	  if (IS_MOVE_OFF_BOARD(m)){

	    printf("move off board\n");

	    continue;

	  }

	  Board tempb = b;
	
	  int num_flip_tmp0 = FlipDisks(m, &tempb, color, 0, 1);


	  /* if no disks have been flipped */ 
	  if (num_flip_tmp0 == 0){

	    continue;

	  }

	  {
	    



	    PlaceOrFlip(m, &tempb, color);
	    ////////////////////////////////////////////////////////////////////////
	    // change to another color;


	    int num_flip_tmp1 = CEnumerateLegalMoves(tempb, OTHERCOLOR(color), bestm, index-1, fix_index);

	    // In order to avoid the case where there are less number left than the depth specified;

	    int num_flip_tmp;
	    if (num_flip_tmp1>20){
	      
	      num_flip_tmp = num_flip_tmp0;
	      
	    }
	    else{

	      num_flip_tmp = num_flip_tmp0 - num_flip_tmp1;


	    }
	      // If the last round, i.e, the number of available move for the step is available, we add the number of moves by one;



	    num_moves++;


	    // Select a better move;
	    if (num_flip_tmp > num_flip){

	      // mark down the best move;
	      *bestm = m;

	      num_flip =  num_flip_tmp;

	    }


	  }

	}
	thisrow >>= 1;
      }
      my_neighbor_moves >>= 8;
    }

    /* For the outest loop, we need to return the num_moves to determine wether to move;*/
    return num_moves;
  }
  // In the middle; 
 else{

    Board neighbors = NeighborMoves(b, color);

    cilk::reducer_max<int> num_flip;

       
    cilk_for(int row=8; row >=1; row--) {
  
      /* Note: for parallization, we need to initialize the reducer at each process, or it will be initialized as zero at the beginning; */

      
      ull my_neighbor_moves = neighbors.disks[color];
      for(int tempi=8; tempi>row; tempi--){

	my_neighbor_moves >>= 8;

      }

      //    printf("num of flip:\t%d\n",num_flip.get_value());

      ull thisrow = my_neighbor_moves & ROW8;


      for(int col=8; thisrow && (col >= 1); col--) {
	//	printf("The row and col searched is %d\t%d\n",row,col);

	if (thisrow & COL8) {
	  Move m = { row, col };
	
	  /* if board position occupied */
	  ull movebit = MOVE_TO_BOARD_BIT(m);

	  if (movebit & (b.disks[X_BLACK] | b.disks[O_WHITE])) {

	    continue;

	  }

	  /* if move is not on the board */
	  if (IS_MOVE_OFF_BOARD(m)){

	    continue;

	  }

	  Board tempb = b;
	  //	    Board *tempbmem = (Board *) malloc(sizeof(b));
	  //	    memcpy(tempbmem,&b,sizeof(b));
	  //	    Board tempb = *tempbmem; 


	  	
	  int num_flip_tmp0 = FlipDisks(m, &tempb, color, 0, 1);

	  //	  printf("num of flip tmp0 is %d\n", num_flip_tmp0);

	  if (num_flip_tmp0 == 0){

	    continue;

	  }

	  {
	    

	    PlaceOrFlip(m, &tempb, color);

	    ////////////////////////////////////////////////////////////////////////
	    // Remember to change to another color;

	    int num_flip_tmp1 = CEnumerateLegalMoves(tempb, OTHERCOLOR(color), bestm, index-1, fix_index);

	    // In order to avoid the case where there are less number left than the depth specified;

	    int num_flip_tmp;
	    if (num_flip_tmp1>20){
	      
	      num_flip_tmp = num_flip_tmp0;
	      
	    }
	    else{

	      num_flip_tmp = num_flip_tmp0 - num_flip_tmp1;


	    }


	    // Select a better move;

	    num_flip = cilk::max_of(num_flip_tmp, num_flip);

	  }
	}
	thisrow >>= 1;
      }
 
    }
  
    /* For the middle one, we need to return the num of flips in order to continue the iteration; */
    return num_flip.get_value();
  }
}


int HumanTurn(Board *b, int color)
{
  Board legal_moves;
  int num_moves = EnumerateLegalMoves(*b, color, &legal_moves);
  if (num_moves > 0) {
    ReadMove(color, b);
    return 1;
  } else return 0;
}


int ComputerTurn(Board *b, int color, int index, int fix_index)
{
  Move bestm;

  int i;  
  int num_moves = CEnumerateLegalMoves(*b, color, &bestm, index, fix_index);
  printf("The num of move is %d\n",num_moves);

  if (num_moves > 0) {

    printf("The move is %d\t%d\n", bestm.row, bestm.col);
    CReadMove(color, b, &bestm);
    return 1;

  }else return 0;

}


int CountBitsOnBoard(Board *b, int color)
{
  ull bits = b->disks[color];
  int ndisks = 0;
  for (; bits ; ndisks++) {
    bits &= bits - 1; // clear the least significant bit set
  }
  return ndisks;
}

void EndGame(Board b)
{
  int o_score = CountBitsOnBoard(&b,O_WHITE);
  int x_score = CountBitsOnBoard(&b,X_BLACK);
  printf("Game over. \n");
  if (o_score == x_score)  {
    printf("Tie game. Each player has %d disks\n", o_score);
  } else { 
    printf("X has %d disks. O has %d disks. %c wins.\n", x_score, o_score, 
	      (x_score > o_score ? 'X' : 'O'));
  }
}

int cilk_main (int argc, char * argv[]) 
{
  // Open the input file;
  char player1[BUFSIZ];
  char player2[BUFSIZ];
  int num_ahead1;
  int num_ahead2;

  FILE *fp;

  fp = fopen("input.txt", "r");

  fgets(player1, BUFSIZ, fp);
  fscanf(fp, "%d\n", &num_ahead1);
  fgets(player2, BUFSIZ, fp);
  fscanf(fp, "%d\n", &num_ahead2);
 
  fclose(fp);

  Board gameboard = start;
  int move_possible;
  int index1, index2;
  PrintBoard(gameboard);

  if (player1[0] == 'c' && player2[0] == 'c'){

    index1 = num_ahead1;
    index2 = num_ahead2;
    do {

	printf("\n ABOUT TO MAKE THE FIRST PLAYER MOVE\n");
      move_possible = 
	ComputerTurn(&gameboard, X_BLACK, index1, index1); 

	printf("\n ABOUT TO MAKE THE SECOND PLAYER MOVE\n");

	move_possible |= 
	ComputerTurn(&gameboard, O_WHITE, index2, index2);

	//	printf("\n DONE WITH PLAYER MOVEs; GOING INTO THE NEXT ITERATION\n");

    } while(move_possible);
  }
  else if (player1[0] == 'h' && player2[0] == 'h'){
    do {
      move_possible = 
  
	HumanTurn(&gameboard, X_BLACK) | 
	HumanTurn(&gameboard, O_WHITE);

    } while(move_possible);
  }
  else if (player1[0] == 'c' && player2[0] == 'h'){

    index1 = num_ahead1;
    do {
      move_possible = 
  
	ComputerTurn(&gameboard, X_BLACK, index1, index1) | 
	HumanTurn(&gameboard, O_WHITE);

    } while(move_possible);
  }
  else if (player1[0] == 'h' && player2[0] == 'c'){

    index2 = num_ahead2;
    do {
      move_possible = 
  
	HumanTurn(&gameboard, X_BLACK) | 
	ComputerTurn(&gameboard, O_WHITE, index2, index2);

    } while(move_possible);
  }
	
  EndGame(gameboard);
	
  return 0;
}
