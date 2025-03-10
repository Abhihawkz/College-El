package recycleright;

import javafx.animation.AnimationTimer;
import javafx.application.Application;
import javafx.scene.Group;
import javafx.scene.Scene;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.image.Image;
import javafx.scene.input.KeyCode;
import javafx.scene.paint.Color;
import javafx.scene.text.Font;
import javafx.scene.text.FontWeight;
import javafx.stage.Stage;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.Random;

public class RecycleRightGame extends Application {

    // Game window dimensions
    private static final int WIDTH = 800;
    private static final int HEIGHT = 600;

    // Game variables
    private int score = 0;
    private int lives = 3;
    private boolean gameOver = false;
    private boolean gameStarted = false;

    // Player variables
    private double playerX = WIDTH / 2 - 40;
    private double playerSpeed = 7.0;
    private Image playerImage;

    // Item variables
    private ArrayList<FallingItem> fallingItems = new ArrayList<>();
    private long lastItemGenTime = 0;
    private Random random = new Random();

    // Images
    private Image backgroundImage;
    private Image wasteItemImage;
    private Image edibleItemImage;
    private Image heartImage;

    public static void main(String[] args) {
        launch(args);
    }

    @Override
    public void start(Stage primaryStage) {
        primaryStage.setTitle("Recycle Right");

        // Load images
        playerImage = new Image(getClass().getResourceAsStream("/resources/bucket.png"), 80, 80, true, true);
        wasteItemImage = new Image(getClass().getResourceAsStream("/resources/waste.png"), 50, 50, true, true);
        edibleItemImage = new Image(getClass().getResourceAsStream("/resources/edible.png"), 50, 50, true, true);
        heartImage = new Image(getClass().getResourceAsStream("/resources/heart.png"), 30, 30, true, true);
        backgroundImage = new Image(getClass().getResourceAsStream("/resources/background.png"), WIDTH, HEIGHT, false, true);

        // Set up game canvas
        Group root = new Group();
        Canvas canvas = new Canvas(WIDTH, HEIGHT);
        root.getChildren().add(canvas);

        Scene scene = new Scene(root);
        primaryStage.setScene(scene);

        GraphicsContext gc = canvas.getGraphicsContext2D();

        // Handle keyboard input
        scene.setOnKeyPressed(e -> {
            if (e.getCode() == KeyCode.LEFT) {
                playerX -= playerSpeed;
                if (playerX < 0)
                    playerX = 0;
            } else if (e.getCode() == KeyCode.RIGHT) {
                playerX += playerSpeed;
                if (playerX > WIDTH - 80)
                    playerX = WIDTH - 80;
            } else if (e.getCode() == KeyCode.SPACE) {
                if (gameOver) {
                    // Reset game
                    resetGame();
                } else if (!gameStarted) {
                    gameStarted = true;
                }
            }
        });

        // Game loop
        new AnimationTimer() {
            @Override
            public void handle(long currentTime) {
                // Clear canvas
                gc.clearRect(0, 0, WIDTH, HEIGHT);

                // Draw background
                gc.drawImage(backgroundImage, 0, 0);

                if (!gameStarted) {
                    drawStartScreen(gc);
                    return;
                }

                if (gameOver) {
                    drawGameOver(gc);
                    return;
                }

                // Generate new items
                if (currentTime - lastItemGenTime > 1_000_000_000) {
                    generateItem();
                    lastItemGenTime = currentTime;
                }

                // Update and draw falling items
                updateItems();
                drawItems(gc);

                // Draw player
                gc.drawImage(playerImage, playerX, HEIGHT - 100);

                // Draw UI elements
                drawUI(gc);

                // Check for collisions
                checkCollisions();

                // Check game over condition
                if (lives <= 0) {
                    gameOver = true;
                }
            }
        }.start();

        primaryStage.show();
    }

    private void resetGame() {
        score = 0;
        lives = 3;
        gameOver = false;
        fallingItems.clear();
        playerX = WIDTH / 2 - 40;
    }

    private void generateItem() {
        double x = random.nextDouble() * (WIDTH - 50);
        double speed = 2.0 + random.nextDouble() * 2.0;
        boolean isWaste = random.nextBoolean();

        fallingItems.add(new FallingItem(x, -50, speed, isWaste));
    }

    private void updateItems() {
        Iterator<FallingItem> iterator = fallingItems.iterator();
        while (iterator.hasNext()) {
            FallingItem item = iterator.next();
            item.y += item.speed;

            // Remove items that fall out of screen
            if (item.y > HEIGHT) {
                iterator.remove();

                // Penalize missing waste items
                if (item.isWaste) {
                    lives--;
                }
            }
        }
    }

    private void drawItems(GraphicsContext gc) {
        for (FallingItem item : fallingItems) {
            Image img = item.isWaste ? wasteItemImage : edibleItemImage;
            gc.drawImage(img, item.x, item.y);
        }
    }

    private void checkCollisions() {
        Iterator<FallingItem> iterator = fallingItems.iterator();
        while (iterator.hasNext()) {
            FallingItem item = iterator.next();

            // Simple collision detection
            if (item.y + 50 > HEIGHT - 100 && item.y < HEIGHT - 20) {
                if (item.x + 50 > playerX && item.x < playerX + 80) {
                    iterator.remove();

                    if (item.isWaste) {
                        // Correct catch: waste item
                        score += 10;
                    } else {
                        // Wrong catch: edible item
                        lives--;
                    }
                }
            }
        }
    }

    private void drawUI(GraphicsContext gc) {
        // Set up text style
        gc.setFill(Color.WHITE);
        gc.setFont(Font.font("Arial", FontWeight.BOLD, 24));

        // Draw score
        gc.fillText("POINTS: " + String.format("%02d", score), 20, 30);

        // Draw lives
        for (int i = 0; i < lives; i++) {
            gc.drawImage(heartImage, WIDTH - 40 - (i * 35), 20);
        }
    }

    private void drawStartScreen(GraphicsContext gc) {
        gc.setFill(Color.WHITE);
        gc.setFont(Font.font("Arial", FontWeight.BOLD, 36));
        gc.fillText("RECYCLE RIGHT", WIDTH / 2 - 150, HEIGHT / 2 - 50);

        gc.setFont(Font.font("Arial", FontWeight.NORMAL, 24));
        gc.fillText("Catch the waste items, avoid the edible ones!", WIDTH / 2 - 250, HEIGHT / 2);
        gc.fillText("Press SPACE to start", WIDTH / 2 - 120, HEIGHT / 2 + 50);
    }

    private void drawGameOver(GraphicsContext gc) {
        gc.setFill(Color.WHITE);
        gc.setFont(Font.font("Arial", FontWeight.BOLD, 36));
        gc.fillText("GAME OVER", WIDTH / 2 - 120, HEIGHT / 2 - 50);

        gc.setFont(Font.font("Arial", FontWeight.NORMAL, 24));
        gc.fillText("Final Score: " + score, WIDTH / 2 - 80, HEIGHT / 2);
        gc.fillText("Press SPACE to restart", WIDTH / 2 - 130, HEIGHT / 2 + 50);
    }

    // Class to represent falling items (waste or edible)
    private class FallingItem {
        double x;
        double y;
        double speed;
        boolean isWaste;

        public FallingItem(double x, double y, double speed, boolean isWaste) {
            this.x = x;
            this.y = y;
            this.speed = speed;
            this.isWaste = isWaste;
        }
    }
}