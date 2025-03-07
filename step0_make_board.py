import cv2
import numpy as np
import yaml
import argparse

def generate_charuco_board(config_file):
    # Load the configuration file
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Read configuration parameters
    squares_x = config.get("squares_x", 5)
    squares_y = config.get("squares_y", 7)
    square_length_mm = config.get("square_length_mm", 30) / 1000.0  # convert to meters
    marker_length_mm = config.get("marker_length_mm", 24) / 1000.0  # convert to meters
    aruco_dict_name = config.get("aruco_dict", "DICT_4X4_50")

    # Get the corresponding ArUco dictionary
    aruco_dict = getattr(cv2.aruco, aruco_dict_name, None)
    if aruco_dict is None:
        raise ValueError(f"Invalid ArUco dictionary: {aruco_dict_name}")

    # Create the ChArUco board 
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict)
    charuco_board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y), square_length_mm, marker_length_mm, aruco_dict
    )
    # Create a blank image to draw the board
    board_image_size = (int(squares_x * square_length_mm), int(squares_y * square_length_mm))
    board_image = np.zeros((board_image_size[1], board_image_size[0]), dtype=np.uint8)

    # Draw the ChArUco board
    charuco_board_image = charuco_board.generateImage(board_image_size,marginSize=5)
    

    # Convert the board to color (so we can mark the origin in red)
    charuco_board_image_color = cv2.cvtColor(charuco_board_image, cv2.COLOR_GRAY2BGR)

    # Mark the origin (top-left corner)
    origin_radius = 2
    origin_color = (0, 0, 0)  # Red in BGR
    origin_thickness = 2
    cv2.circle(charuco_board_image_color, (0, 0), origin_radius, origin_color, origin_thickness)

    # # Add text for the origin
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # font_scale = 0.8
    # font_thickness = 2
    # text_color = (0, 0, 255)  # Red in BGR
    # cv2.putText(
    #     charuco_board_image_color, "Origin (0,0)", (20, 40), font, font_scale, text_color, font_thickness
    # )

    return charuco_board_image_color

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate a ChArUco board as a PNG image.")
    parser.add_argument("--board", type=str, required=True, help="Path to the board YAML configuration file.")
    args = parser.parse_args()

    # Generate the ChArUco board
    try:
        charuco_image = generate_charuco_board(args.board)
        # Save the PNG image
        output_file = "charuco_board.png"
        cv2.imwrite(output_file, charuco_image)
        print(f"ChArUco board saved to {output_file}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()