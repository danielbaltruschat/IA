package uk.ac.cam.cl.gfxintro.dntb2.tick1;

public class Sphere extends SceneObject {

	// Sphere coefficients
	private final double SPHERE_KD = 0.8;
	private final double SPHERE_KS = 1.2;
	private final double SPHERE_ALPHA = 10;
	private final double SPHERE_REFLECTIVITY = 0.3;

	// The world-space position of the sphere
	protected Vector3 position;

	public Vector3 getPosition() {
		return position;
	}

	// The radius of the sphere in world units
	private double radius;

	public Sphere(Vector3 position, double radius, ColorRGB colour) {
		this.position = position;
		this.radius = radius;
		this.colour = colour;

		this.phong_kD = SPHERE_KD;
		this.phong_kS = SPHERE_KS;
		this.phong_alpha = SPHERE_ALPHA;
		this.reflectivity = SPHERE_REFLECTIVITY;
	}

	public Sphere(Vector3 position, double radius, ColorRGB colour, double kD, double kS, double alphaS, double reflectivity, ColorRGB transmittance) {
		this.position = position;
		this.radius = radius;
		this.colour = colour;

		this.phong_kD = kD;
		this.phong_kS = kS;
		this.phong_alpha = alphaS;
		this.reflectivity = reflectivity;
		this.transmittance = transmittance;
	}

	/*
	 * Calculate intersection of the sphere with the ray. If the ray starts inside the sphere,
	 * intersection with the surface is also found.     
	 */
	public RaycastHit intersectionWith(Ray ray) {

		// Get ray parameters
		Vector3 O = ray.getOrigin();
		Vector3 D = ray.getDirection();
		
		// Get sphere parameters
		Vector3 C = position;
		double r = radius;

		// Calculate quadratic coefficients
		double a = D.dot(D);
		double b = 2 * D.dot(O.subtract(C));
		double c = (O.subtract(C)).dot(O.subtract(C)) - Math.pow(r, 2);

		double dSquared = b*b - 4*a*c;

		// If no intersection return empty RayCastHit object
		if (dSquared < 0) return new RaycastHit();
		else {
			double d = Math.sqrt(dSquared);
			double s1 = (d-b) / 2*a;
			double s2 = (-d-b) / 2*a;
			double s;
			if (s1 < 0 && s2 < 0) {
				return new RaycastHit();
			}
			else if (s1 < 0) {
				s = s2;
			}
			else if (s2 < 0) {
				s = s1;
			}
			else {
				s = Math.min(s1, s2);
			}
			Vector3 pos = O.add(D.scale(s));
			return new RaycastHit(this, s * D.magnitude(), pos, getNormalAt(pos));
		}

		// TODO: Determine if ray and sphere intersect - if not return an empty RaycastHit
        // TODO: If so, work out any point of intersection
        // TODO: Then return a RaycastHit that includes the object, ray distance, point, and normal vector

	}

	// Get normal to surface at position
	public Vector3 getNormalAt(Vector3 position) {
		return position.subtract(this.position).normalised();
	}
}
