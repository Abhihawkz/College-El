import javafx.application.Application;
import javafx.stage.Stage;
import javafx.scene.Scene;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.image.Image;
import javafx.scene.layout.HBox;
import javafx.scene.layout.StackPane;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.scene.text.Font;
import javafx.scene.text.FontWeight;
import javafx.scene.text.Text;
import javafx.scene.control.Button;
import javafx.scene.input.KeyCode;
import javafx.animation.AnimationTimer;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.Random;

public class WasteCollectorGame extends Application {
    private static final int WIDTH = 600;
    private static final int HEIGHT = 800;

    private Image playerImage = new Image("bucket.png"); // Ensure this has a transparent background
    private Image wasteImage = new Image("waste.png");  // Ensure this has a transparent background
    private Image goodImage = new Image("tomato.png");  // Ensure this has a transparent background
    private Image heartImage = new Image("heart.png");  // Heart icon for lives

    private double playerX = WIDTH / 2 - 50;
    private final double playerY = HEIGHT - 120;
    private final double playerSpeed = 8;

    private boolean moveLeft = false, moveRight = false;
    private int score = 0, lives = 3;
    private boolean gameOver = false;

    private ArrayList<FallingObject> fallingObjects = new ArrayList<>();
    private Random random = new Random();

    private AnimationTimer gameLoop;
    private Stage mainStage;
    private Scene gameScene;
    private StackPane root;

    @Override
    public void start(Stage stage) {
        this.mainStage = stage;
        root = new StackPane();
        Canvas canvas = new Canvas(WIDTH, HEIGHT);
        GraphicsContext gc = canvas.getGraphicsContext2D();
        root.getChildren().add(canvas);
        gameScene = new Scene(root);

        Text gameTitle = new Text("RecycleRight");
        gameTitle.setFont(Font.font("Arial", FontWeight.BOLD, 40));
        gameTitle.setFill(Color.LIMEGREEN);
        gameTitle.setStroke(Color.DARKGREEN);
        gameTitle.setStrokeWidth(2);
        gameTitle.setTranslateY(-HEIGHT / 2 + 50); 

   
        HBox topBar = new HBox();
        topBar.setSpacing(20);
        topBar.setStyle("-fx-alignment: center; -fx-padding: 10px;");

        Text scoreText = new Text("Score: 0");
        scoreText.setFont(Font.font("Verdana", FontWeight.BOLD, 24));
        scoreText.setFill(Color.WHITE);

        HBox livesBox = new HBox();
        livesBox.setSpacing(5);
        updateLivesDisplay(livesBox); 

        topBar.getChildren().addAll(scoreText, gameTitle, livesBox);
        root.getChildren().add(topBar);

        gameScene.setOnKeyPressed(e -> {
            if (e.getCode() == KeyCode.LEFT) moveLeft = true;
            if (e.getCode() == KeyCode.RIGHT) moveRight = true;
        });

        gameScene.setOnKeyReleased(e -> {
            if (e.getCode() == KeyCode.LEFT) moveLeft = false;
            if (e.getCode() == KeyCode.RIGHT) moveRight = false;
        });

        stage.setScene(gameScene);
        stage.setTitle("RecycleRight - Waste Collector Game");
        stage.show();

        startGameLoop(gc);
    }

    private void startGameLoop(GraphicsContext gc) {
        gameLoop = new AnimationTimer() {
            @Override
            public void handle(long now) {
                update();
                draw(gc);
            }
        };
        gameLoop.start();
    }

    private void update() {
        if (gameOver) return;

        if (moveLeft && playerX > 0) playerX -= playerSpeed;
        if (moveRight && playerX < WIDTH - 100) playerX += playerSpeed;

        if (random.nextInt(100) < 2) {
            boolean isWaste = random.nextBoolean();
            fallingObjects.add(new FallingObject(random.nextInt(WIDTH - 50), 0, isWaste));
        }

        Iterator<FallingObject> iterator = fallingObjects.iterator();
        while (iterator.hasNext()) {
            FallingObject obj = iterator.next();
            obj.y += 5;

            if (obj.y + 50 >= playerY && obj.x > playerX && obj.x < playerX + 100) {
                if (obj.isWaste) {
                    score += 10;
                } else {
                    lives--;
                }
                iterator.remove();
            }

            if (obj.y > HEIGHT) {
                iterator.remove();
            }
        }

        if (lives <= 0) {
            gameOver = true;
            gameLoop.stop();
            showGameOverScreen();
        }
    }

    private void draw(GraphicsContext gc) {
        gc.clearRect(0, 0, WIDTH, HEIGHT);

        gc.setFill(Color.DARKSLATEGRAY);
        gc.fillRect(0, 0, WIDTH, HEIGHT);

        gc.drawImage(playerImage, playerX, playerY, 100, 100);
        for (FallingObject obj : fallingObjects) {
            gc.drawImage(obj.isWaste ? wasteImage : goodImage, obj.x, obj.y, 50, 50);
        }
    }

    private void updateLivesDisplay(HBox livesBox) {
        livesBox.getChildren().clear();
        for (int i = 0; i < lives; i++) {
            javafx.scene.image.ImageView heartView = new javafx.scene.image.ImageView(heartImage);
            heartView.setFitWidth(30);
            heartView.setFitHeight(30);
            livesBox.getChildren().add(heartView);
        }
    }

    private void showGameOverScreen() {
        VBox gameOverPane = new VBox(20);
        gameOverPane.setStyle("-fx-alignment: center; -fx-background-color: rgba(0,0,0,0.7); -fx-padding: 40px;");

        Text gameOverText = new Text("GAME OVER");
        gameOverText.setFont(Font.font("Arial", FontWeight.BOLD, 50));
        gameOverText.setFill(Color.RED);

        Text scoreText = new Text("Final Score: " + score);
        scoreText.setFont(Font.font("Arial", FontWeight.BOLD, 30));
        scoreText.setFill(Color.WHITE);

        Button restartButton = new Button("Restart");
        restartButton.setStyle("-fx-font-size: 20px; -fx-background-color: orange; -fx-text-fill: black; -fx-padding: 10px;");
        restartButton.setOnAction(e -> restartGame());

        gameOverPane.getChildren().addAll(gameOverText, scoreText, restartButton);
        root.getChildren().add(gameOverPane);
    }

    private void restartGame() {
        score = 0;
        lives = 3;
        gameOver = false;
        playerX = WIDTH / 2 - 50;
        fallingObjects.clear();
        root.getChildren().clear();
        Canvas canvas = new Canvas(WIDTH, HEIGHT);
        GraphicsContext gc = canvas.getGraphicsContext2D();
        root.getChildren().add(canvas);

        HBox topBar = new HBox();
        topBar.setSpacing(20);
        topBar.setStyle("-fx-alignment: center; -fx-padding: 10px;");

        Text scoreText = new Text("Score: 0");
        scoreText.setFont(Font.font("Verdana", FontWeight.BOLD, 24));
        scoreText.setFill(Color.WHITE);

        Text gameTitle = new Text("RecycleRight");
        gameTitle.setFont(Font.font("Arial", FontWeight.BOLD, 40));
        gameTitle.setFill(Color.LIMEGREEN);
        gameTitle.setStroke(Color.DARKGREEN);
        gameTitle.setStrokeWidth(2);

        HBox livesBox = new HBox();
        livesBox.setSpacing(5);
        updateLivesDisplay(livesBox);

        topBar.getChildren().addAll(scoreText, gameTitle, livesBox);
        root.getChildren().add(topBar);

        startGameLoop(gc);
    }

    class FallingObject {
        double x, y;
        boolean isWaste;

        FallingObject(double x, double y, boolean isWaste) {
            this.x = x;
            this.y = y;
            this.isWaste = isWaste;
        }
    }

    public static void main(String[] args) {
        launch();
    }
}