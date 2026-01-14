import requests
import os
import cv2
import numpy as np
from collections import Counter

def analyze_tongue_image_local(segmented_image_path, original_image_path=None):
    """
    Local image analysis based on color, texture, and visual features.
    Provides health insights without requiring external API.
    """
    img = cv2.imread(segmented_image_path)
    if img is None:
        return "Error: Could not read segmented image."
    
    # Convert to different color spaces for analysis
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Create mask of non-black pixels (actual tongue area)
    # Use a very low threshold to catch any visible pixels
    mask = cv2.inRange(img_gray, 5, 255)
    tongue_pixels = img_rgb[mask > 0]
    
    # If segmented image has no visible tongue area, try using original image
    if len(tongue_pixels) < 100:  # Less than 100 pixels detected
        if original_image_path and os.path.exists(original_image_path):
            print("Segmented image has minimal content, analyzing original image instead...")
            img = cv2.imread(original_image_path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Use entire image for analysis
                mask = np.ones(img_gray.shape, dtype=np.uint8) * 255
                tongue_pixels = img_rgb.reshape(-1, 3)
        else:
            # If still no pixels, analyze the entire segmented image anyway
            print("Warning: Using entire image for analysis as segmentation may not be accurate")
            mask = np.ones(img_gray.shape, dtype=np.uint8) * 255
            tongue_pixels = img_rgb.reshape(-1, 3)
    
    if len(tongue_pixels) == 0:
        return "Error: Could not analyze image. Please ensure the image contains a visible tongue."
    
    # Analyze color characteristics with more detailed metrics
    avg_color = np.mean(tongue_pixels, axis=0)
    r, g, b = avg_color
    
    # Calculate color ratios for better analysis
    total_intensity = r + g + b
    r_ratio = r / total_intensity if total_intensity > 0 else 0
    g_ratio = g / total_intensity if total_intensity > 0 else 0
    b_ratio = b / total_intensity if total_intensity > 0 else 0
    
    # Analyze coating (white/yellow areas) with better detection
    hsv_pixels = img_hsv[mask > 0]
    saturation = np.mean(hsv_pixels[:, 1])
    value = np.mean(hsv_pixels[:, 2])
    hue = np.mean(hsv_pixels[:, 0])
    
    # Better white coating detection - check for light colors
    white_coating = np.sum((tongue_pixels[:, 0] > 180) & (tongue_pixels[:, 1] > 180) & (tongue_pixels[:, 2] > 180))
    coating_percentage = (white_coating / len(tongue_pixels)) * 100
    
    # Better yellow coating detection - check HSV hue range for yellow (20-30)
    yellow_mask = (hsv_pixels[:, 0] >= 15) & (hsv_pixels[:, 0] <= 35) & (hsv_pixels[:, 1] > 50)
    yellow_coating = np.sum(yellow_mask)
    yellow_percentage = (yellow_coating / len(tongue_pixels)) * 100
    
    # Detect redness (inflammation indicators) - more nuanced
    redness = np.mean(tongue_pixels[:, 0])
    redness_std = np.std(tongue_pixels[:, 0])
    
    # Detect dryness (texture analysis using variance and other metrics)
    gray_tongue = img_gray[mask > 0]
    texture_variance = np.var(gray_tongue)
    texture_mean = np.mean(gray_tongue)
    
    # Detect cracks (edge detection with adaptive thresholds)
    # Use adaptive thresholding for better edge detection
    adaptive_thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    edges = cv2.Canny(img_gray, 30, 100)
    edge_density = np.sum(edges > 0) / (img_gray.shape[0] * img_gray.shape[1])
    
    # Detect patches/spots (areas of different color)
    color_variance = np.var(tongue_pixels, axis=0)
    color_variance_total = np.sum(color_variance)
    
    # Build analysis report with more nuanced conditions
    findings = []
    conditions = []
    recommendations = []
    
    # Enhanced Color analysis with more ranges
    if r > 200 and r_ratio > 0.45 and g < 140:
        findings.append("• Bright red/strawberry tongue color detected")
        conditions.append("Possible fever, inflammation, or infection")
        recommendations.append("Monitor body temperature, rest, and consider consulting a doctor if fever persists")
    elif r > 170 and r_ratio > 0.40:
        findings.append("• Reddish tongue color detected")
        conditions.append("Possible mild inflammation or increased body heat")
        recommendations.append("Stay hydrated, rest adequately, and monitor symptoms")
    elif r < 90 and g < 90 and b < 90:
        findings.append("• Very pale tongue color detected")
        conditions.append("Possible anemia, low energy, or circulation issues")
        recommendations.append("Consider checking iron levels, maintain balanced nutrition, and consult healthcare provider")
    elif r < 120 and g < 120:
        findings.append("• Pale tongue color detected")
        conditions.append("Possible mild anemia or low energy")
        recommendations.append("Ensure adequate iron intake and maintain balanced diet")
    elif r > 140 and r < 170 and g > 100 and g < 140:
        findings.append("• Normal pinkish-red tongue color")
        findings.append("• Healthy tongue coloration observed")
    else:
        findings.append(f"• Tongue color analysis: RGB({int(r)}, {int(g)}, {int(b)})")
        if r_ratio > 0.38:
            findings.append("• Slightly elevated red tone")
    
    # Enhanced Coating analysis
    if coating_percentage > 40:
        findings.append(f"• Heavy white coating detected ({coating_percentage:.1f}% coverage)")
        conditions.append("Possible digestive issues, oral thrush, or yeast infection")
        recommendations.append("Maintain strict oral hygiene, consider antifungal treatment, and review diet")
    elif coating_percentage > 25:
        findings.append(f"• Moderate white coating detected ({coating_percentage:.1f}% coverage)")
        conditions.append("Possible digestive concerns or mild oral issues")
        recommendations.append("Improve oral hygiene routine and consider dietary adjustments")
    elif coating_percentage > 12:
        findings.append(f"• Light white coating detected ({coating_percentage:.1f}% coverage)")
        conditions.append("Mild digestive concerns")
        recommendations.append("Stay hydrated, maintain good oral hygiene, and eat balanced meals")
    elif coating_percentage > 5:
        findings.append(f"• Minimal white coating ({coating_percentage:.1f}% coverage)")
        findings.append("• Generally healthy tongue appearance")
    else:
        findings.append("• Clean tongue surface with minimal coating")
    
    # Yellow coating analysis
    if yellow_percentage > 20:
        findings.append(f"• Significant yellow coating detected ({yellow_percentage:.1f}% coverage)")
        conditions.append("Possible heat-related conditions, digestive issues, or liver concerns")
        recommendations.append("Increase water intake, consume cooling foods, and consider liver health")
    elif yellow_percentage > 10:
        findings.append(f"• Light yellow coating detected ({yellow_percentage:.1f}% coverage)")
        conditions.append("Possible mild heat-related conditions")
        recommendations.append("Stay well-hydrated and avoid excessive spicy/heating foods")
    elif yellow_percentage > 3:
        findings.append(f"• Slight yellowish tint ({yellow_percentage:.1f}% coverage)")
    
    # Enhanced Dryness and Texture analysis
    if texture_variance < 300:
        findings.append("• Very smooth texture detected (severe dryness)")
        conditions.append("Significant dehydration")
        recommendations.append("Immediately increase water intake (10-12 glasses daily) and monitor hydration")
    elif texture_variance < 600:
        findings.append("• Smooth texture detected (moderate dryness)")
        conditions.append("Possible dehydration")
        recommendations.append("Increase daily water intake (aim for 8-10 glasses) and reduce caffeine/alcohol")
    elif texture_variance > 3000:
        findings.append("• Very rough texture detected")
        conditions.append("Possible severe dehydration or nutritional deficiency")
        recommendations.append("Stay well-hydrated, ensure adequate vitamin intake (especially B-complex), and consult healthcare provider")
    elif texture_variance > 1800:
        findings.append("• Rough texture detected")
        conditions.append("Possible dehydration or nutritional deficiency")
        recommendations.append("Increase hydration and ensure adequate vitamin intake")
    elif texture_variance > 800:
        findings.append("• Normal tongue texture")
    else:
        findings.append("• Slightly smooth texture")
    
    # Enhanced Crack detection
    if edge_density > 0.25:
        findings.append("• Multiple cracks/fissures detected (severe)")
        conditions.append("Possible severe dehydration or significant nutritional deficiency")
        recommendations.append("Increase hydration significantly, consider B-vitamin supplements, and consult healthcare provider")
    elif edge_density > 0.18:
        findings.append("• Visible cracks or fissures detected")
        conditions.append("Possible dehydration or nutritional deficiency")
        recommendations.append("Increase hydration and consider B-vitamin supplements")
    elif edge_density > 0.12:
        findings.append("• Minor surface irregularities detected")
        recommendations.append("Maintain adequate hydration")
    
    # Enhanced moisture assessment
    if value < 80:
        findings.append("• Very low brightness/moisture detected")
        conditions.append("Severe dehydration")
        recommendations.append("Drink water immediately and throughout the day, avoid dehydrating beverages")
    elif value < 110:
        findings.append("• Low brightness/moisture detected")
        conditions.append("Dehydration")
        recommendations.append("Increase water intake significantly (8-10 glasses daily)")
    elif value > 180:
        findings.append("• Good moisture levels detected")
    elif value > 140:
        findings.append("• Adequate moisture levels")
    
    # Additional analysis based on saturation
    if saturation < 30:
        findings.append("• Low color saturation (possible coating or dryness)")
    elif saturation > 80:
        findings.append("• High color saturation (vibrant tongue appearance)")
    
    # Color variance analysis for patches
    if color_variance_total > 5000:
        findings.append("• Uneven coloration detected (possible patches or spots)")
        conditions.append("Possible oral health concerns or localized issues")
        recommendations.append("Monitor for changes and consider dental/medical consultation")
    
    # Build final report
    report = "VISUAL ANALYSIS FINDINGS:\n\n"
    report += "Tongue Characteristics:\n"
    report += "\n".join(findings)
    
    if conditions:
        report += "\n\nPOSSIBLE CONDITIONS INDICATED:\n"
        report += "\n".join([f"• {c}" for c in conditions])
    
    if recommendations:
        report += "\n\nRECOMMENDATIONS:\n"
        report += "\n".join([f"• {r}" for r in recommendations])
    
    report += "\n\n⚠️ IMPORTANT DISCLAIMER:\n"
    report += "This analysis is based on visual characteristics only and is NOT a medical diagnosis. "
    report += "If you experience persistent symptoms or concerns, please consult with a qualified healthcare professional."
    
    return report


def analyze_tongue_image(segmented_image_path, original_image_path=None):
    """
    Main function that tries API first, then falls back to local analysis.
    """
    if not os.path.exists(segmented_image_path):
        return "Error: Segmented image file not found."

    # Try OpenAI Vision API if API key is available
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key:
        try:
            import base64
            # Use original image if available, otherwise segmented
            image_to_analyze = original_image_path if original_image_path and os.path.exists(original_image_path) else segmented_image_path
            with open(image_to_analyze, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            headers = {
                "Authorization": f"Bearer {openai_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "gpt-4-vision-preview",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Analyze this human tongue image. "
                                    "Describe visible characteristics such as color, coating, dryness, cracks, or patches. "
                                    "Based on visual appearance, suggest possible general conditions like dehydration, "
                                    "fever, nutritional deficiency, or infection. "
                                    "Provide specific findings and recommendations. "
                                    "Add a clear disclaimer that this is not a medical diagnosis."
                                )
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_data}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 500
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"OpenAI API error: {e}, falling back to local analysis")
    
    # Fallback to local analysis
    return analyze_tongue_image_local(segmented_image_path, original_image_path)
