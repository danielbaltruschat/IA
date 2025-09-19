package uk.ac.cam.cl.gfxintro.dntb2.tick1;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import javax.imageio.ImageIO;

public class Tick1 {
	// Default input and output files
	public static final String DEFAULT_INPUT = "tick1.xml";
	public static final String DEFAULT_OUTPUT = "output.png";
	
	public static final int DEFAULT_BOUNCES = 2; // Default number of ray bounces

	// Height and width of the output image
	private static final int DEFAULT_WIDTH_PX = 800;
	private static final int DEFAULT_HEIGHT_PX = 600;

	public static void usageError() { // Usage information 
		System.err.println("USAGE: <tick2> [--input INPUT] [--output OUTPUT] [--bounces BOUNCES] [--resolution WIDTHxHEIGHT]");
		System.exit(-1);
	}

	public static void main(String[] args) throws IOException {
		// We should have an even number of arguments - each option and its value
		if (args.length % 2 != 0) {
			usageError();
		}
		
		// Parse the input and output filenames from the arguments
		String inputSceneFile = DEFAULT_INPUT, output = DEFAULT_OUTPUT;
		int width = DEFAULT_WIDTH_PX, height = DEFAULT_HEIGHT_PX;

		int bounces = DEFAULT_BOUNCES;
		for (int i = 0; i < args.length; i += 2) {
			switch (args[i]) {
			case "-i":
			case "--input":
				inputSceneFile = args[i + 1];
				break;
			case "-o":
			case "--output":
				output = args[i + 1];
				break;
			case "-b":
			case "--bounces":
				bounces = Integer.parseInt(args[i + 1]);
				break;
			case "-r":
			case "--resolution":
				Pattern res_pat = Pattern.compile("(\\d+)x(\\d+)");
				Matcher m = res_pat.matcher(args[i+1]);
				if( m.find() ) {
					width = Integer.parseInt(m.group(1));
					height = Integer.parseInt(m.group(2));
					if( width <= 0 || height <= 0 || width > 4096 || height >= 4096 ) {
						System.err.println("unsupported resolution: " + args[i + 1]);
						usageError();
					}
				} else {
					System.err.println("unsupported resolution: " + args[i + 1]);
					usageError();
				}
				break;
			default:
				System.err.println("Unknown option: " + args[i]);
				usageError();
			}
		}

		// Create the scene from the XML file
		System.out.printf( "Loading scene '%s'\n", inputSceneFile );
		Scene scene = new SceneLoader(inputSceneFile).getScene();
		
		// Create the image and colour the pixels
		BufferedImage image = new Renderer(width, height, bounces).render(scene);
		
		// Save the image to disk
		File save = new File(output);
		ImageIO.write(image, "png", save);
	}
}
