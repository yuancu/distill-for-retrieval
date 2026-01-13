"""Create a routed retrieval model that can be loaded with SentenceTransformer.

Usage:
    # Create model
    python create_routed_model.py student_path teacher_path threshold output_path

    # Load with standard SentenceTransformer API
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('output_path')
    embeddings = model.encode(['text1', 'text2'])

    # Access routing decisions
    routing = model[2].last_routing_decisions  # Index 2 is the RoutingModule

    # Adjust threshold
    model[2].threshold = 0.7
"""

import os
import json
import argparse
from sentence_transformers import SentenceTransformer, models
import torch
from typing import Optional

from distill.models import Router
from distill.models import RoutingModule


def create_routed_model(
    student_name_or_path: str,
    teacher_name_or_path: str,
    threshold: float,
    output_path: str,
    device: Optional[str] = None
):
    """Create a routed retrieval model that can be loaded with SentenceTransformer.

    The resulting model structure:
        0. Transformer (student)
        1. Pooling (student)
        2. RoutingModule (contains teacher + router)
        3. Normalize

    Args:
        student_name_or_path: Path or name of student model (must contain router.pt or router.safetensors)
        teacher_name_or_path: Path or name of teacher model
        threshold: Routing threshold (0-1). If difficulty > threshold, use teacher
        output_path: Directory to save the routed model

    Returns:
        Path to saved model
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 80)
    print("Creating Routed SentenceTransformer Model")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  - Student: {student_name_or_path}")
    print(f"  - Teacher: {teacher_name_or_path}")
    print(f"  - Threshold: {threshold:.2f}")
    print(f"  - Output: {output_path}")
    print(f"  - Device: {device}")

    # Step 1: Load student model
    print(f"\n1. Loading student model...")
    student_model = SentenceTransformer(student_name_or_path, device=device)
    student_dim = student_model.get_sentence_embedding_dimension()
    print(f"   ✓ Student loaded (dimension: {student_dim}d)")

    # Step 2: Load teacher model
    print(f"\n2. Loading teacher model...")
    teacher_model = SentenceTransformer(teacher_name_or_path, device=device)
    teacher_dim = teacher_model.get_sentence_embedding_dimension()
    print(f"   ✓ Teacher loaded (dimension: {teacher_dim}d → {student_dim}d truncated)")

    # Step 3: Load router
    print(f"\n3. Loading router...")
    router_path = os.path.join(student_name_or_path, "router.safetensors")
    if not os.path.exists(router_path):
        router_path = os.path.join(student_name_or_path, "router.pt")

    if not os.path.exists(router_path):
        raise FileNotFoundError(
            f"Router not found at {student_name_or_path}/router.safetensors or router.pt\n"
            f"Make sure the student model was trained with 'train_router: true' in Phase 2"
        )

    # Infer router architecture from student model config
    config_path = os.path.join(student_name_or_path, "0_Transformer", "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            transformer_config = json.load(f)
            base_student_dim = transformer_config.get('hidden_size', student_dim)
    else:
        base_student_dim = student_dim

    # Create router
    router_hidden_dim = 256  # Default from models.py
    router = Router(student_dim=base_student_dim, hidden_dim=router_hidden_dim)

    # Load router weights
    if router_path.endswith('.safetensors'):
        try:
            from safetensors.torch import load_file
            state_dict = load_file(router_path, device=str(device))
        except ImportError:
            raise ImportError("safetensors not installed. Install with: pip install safetensors")
    else:
        state_dict = torch.load(router_path, map_location=device, weights_only=True)

    router.load_state_dict(state_dict)
    router.to(device)
    print(f"   ✓ Router loaded from {os.path.basename(router_path)}")
    print(f"   ✓ Router architecture: {base_student_dim}d → {router_hidden_dim}d → 1")

    # Step 4: Create routing module
    print(f"\n4. Creating routing module...")
    routing_module = RoutingModule(
        teacher_model=teacher_model,
        router=router,
        threshold=threshold,
        student_dim=student_dim,
        router_hidden_dim=router_hidden_dim
    )
    print(f"   ✓ Routing module created")

    # Step 5: Build complete SentenceTransformer with routing
    print(f"\n5. Building SentenceTransformer with routing...")

    # Extract student modules
    transformer = student_model[0]  # Transformer
    pooling = student_model[1]      # Pooling

    # Build module list: Transformer → Pooling → RoutingModule → Normalize
    modules = [
        transformer,
        pooling,
        routing_module,
        models.Normalize()  # Normalize after routing
    ]

    # Create the complete model
    routed_model = SentenceTransformer(modules=modules, device=device)
    print(f"   ✓ Model structure:")
    print(f"      [0] Transformer (student)")
    print(f"      [1] Pooling (mean)")
    print(f"      [2] RoutingModule (student/teacher routing)")
    print(f"      [3] Normalize (L2)")

    # Step 6: Test the model
    print(f"\n6. Testing model with diverse MSMARCO samples...")

    # 50 diverse samples from MSMARCO corpus
    test_sentences = [
        "The presence of communication amid scientific minds was equally important to the success of the Manhattan Project as scientific intellect was. The only cloud hanging over the impressive achievement of the atomic researchers and engineers is what their success truly meant; hundreds of thousands of innocent lives obliterated.",
        "The Manhattan Project and its atomic bomb helped bring an end to World War II. Its legacy of peaceful uses of atomic energy continues to have an impact on history and science.",
        "Essay on The Manhattan Project - The Manhattan Project The Manhattan Project was to see if making an atomic bomb possible. The success of this project would forever change the world forever making it known that something this powerful can be manmade.",
        "The Manhattan Project was the name for a project conducted during World War II, to develop the first atomic bomb. It refers specifically to the period of the project from 194 â¦ 2-1946 under the control of the U.S. Army Corps of Engineers, under the administration of General Leslie R. Groves.",
        "versions of each volume as well as complementary websites. The first websiteâThe Manhattan Project: An Interactive Historyâis available on the Office of History and Heritage Resources website, http://www.cfo. doe.gov/me70/history. The Office of History and Heritage Resources and the National Nuclear Security",
        "The Manhattan Project. This once classified photograph features the first atomic bomb â a weapon that atomic scientists had nicknamed Gadget.. The nuclear age began on July 16, 1945, when it was detonated in the New Mexico desert.",
        "Nor will it attempt to substitute for the extraordinarily rich literature on the atomic bombs and the end of World War II. This collection does not attempt to document the origins and development of the Manhattan Project.",
        "Manhattan Project. The Manhattan Project was a research and development undertaking during World War II that produced the first nuclear weapons. It was led by the United States with the support of the United Kingdom and Canada. From 1942 to 1946, the project was under the direction of Major General Leslie Groves of the U.S. Army Corps of Engineers. Nuclear physicist Robert Oppenheimer was the director of the Los Alamos Laboratory that designed the actual bombs. The Army component of the project was designated the",
        "In June 1942, the United States Army Corps of Engineersbegan the Manhattan Project- The secret name for the 2 atomic bombs.",
        "One of the main reasons Hanford was selected as a site for the Manhattan Project's B Reactor was its proximity to the Columbia River, the largest river flowing into the Pacific Ocean from the North American coast.",
        "group discussions, community boards or panels with a third party, or victim and offender dialogues, and requires a skilled facilitator who also has sufficient understanding of sexual assault, domestic violence, and dating violence, as well as trauma and safety issues.",
        "punishment designed to repair the damage done to the victim and community by an offender's criminal act. Ex: community service, Big Brother program indeterminate sentence",
        "Tutorial: Introduction to Restorative Justice. Restorative justice is a theory of justice that emphasizes repairing the harm caused by criminal behaviour. It is best accomplished through cooperative processes that include all stakeholders. This can lead to transformation of people, relationships and communities. Practices and programs reflecting restorative purposes will respond to crime by: 1  identifying and taking steps to repair harm, 2  involving all stakeholders, and. 3  transforming the traditional relationship between communities and their governments in responding to crime.",
        "Organize volunteer community panels, boards, or committees that meet with the offender to discuss the incident and offender obligation to repair the harm to victims and community members. Facilitate the process of apologies to victims and communities. Invite local victim advocates to provide ongoing victim-awareness training for probation staff.",
        "The purpose of this paper is to point out a number of unresolved issues in the criminal justice system, present the underlying principles of restorative justice, and then to review the growing amount of empirical data on victim-offender mediation.",
        "Each of these types of communitiesâthe geographic community of the victim, offender, or crime; the community of care; and civil societyâmay be injured by crime in different ways and degrees, but all will be affected in common ways as well: The sense of safety and confidence of their members is threatened, order within the community is threatened, and (depending on the kind of crime) common values of the community are challenged and perhaps eroded.",
        "The approach is based on a theory of justice that considers crime and wrongdoing to be an offense against an individual or community, rather than the State. Restorative justice that fosters dialogue between victim and offender has shown the highest rates of victim satisfaction and offender accountability.",
        "Inherent in many peopleâs understanding of the notion of ADR is the existence of a dispute between identifiable parties. Criminal justice, however, is not usually conceptualised as a dispute between victim and offender, but is instead seen as a matter concerning the relationship between the offender and the state. This raises a complex question as to whether a criminal offence can properly be described as a âdisputeâ.",
        "Criminal justice, however, is not usually conceptualised as a dispute between victim and offender, but is instead seen as a matter concerning the relationship between the offender and the state. 3 This raises a complex question as to whether a criminal offence can properly be described as a âdisputeâ.",
        "The circle includes a wide range of participants including not only the offender and the victim but also friends and families, community members, and justice system representatives. The primary distinction between conferencing and circles is that circles do not focus exclusively on the offense and do not limit their solutions to repairing the harm between the victim and the offender.",
        "Phloem is a conductive (or vascular) tissue found in plants. Phloem carries the products of photosynthesis (sucrose and glucose) from the leaves to other parts of the plant. â¦ The corresponding system that circulates water and minerals from the roots is called the xylem.",
        "Phloem and xylem are complex tissues that perform transportation of food and water in a plant. They are the vascular tissues of the plant and together form vascular bundles. They work together as a unit to bring about effective transportation of food, nutrients, minerals and water.",
        "Phloem and xylem are complex tissues that perform transportation of food and water in a plant. They are the vascular tissues of the plant and together form vascular bundles.",
        "Phloem is a conductive (or vascular) tissue found in plants. Phloem carries the products of photosynthesis (sucrose and glucose) from the leaves to other parts of the plant.",
        "Unlike xylem (which is composed primarily of dead cells), the phloem is composed of still-living cells that transport sap. The sap is a water-based solution, but rich in sugars made by the photosynthetic areas.",
        "In xylem vessels water travels by bulk flow rather than cell diffusion. In phloem, concentration of organic substance inside a phloem cell (e.g., leaf) creates a diffusion gradient by which water flows into cells and phloem sap moves from source of organic substance to sugar sinks by turgor pressure.",
        "The mechanism by which sugars are transported through the phloem, from sources to sinks, is called pressure flow. At the sources (usually the leaves), sugar molecules are moved into the sieve elements (phloem cells) through active transport.",
        "Phloem carries the products of photosynthesis (sucrose and glucose) from the leaves to other parts of the plant. â¦ The corresponding system that circulates water and minerals from the roots is called the xylem.",
        "Xylem transports water and soluble mineral nutrients from roots to various parts of the plant. It is responsible for replacing water lost through transpiration and photosynthesis. Phloem translocates sugars made by photosynthetic areas of plants to storage organs like roots, tubers or bulbs.",
        "At this time the Industrial Workers of the World had a membership of over 100,000 members. In 1913 William Haywood replaced Vincent Saint John as secretary-treasurer of the Industrial Workers of the World. By this time, the IWW had 100,000 members.",
        "This was not true of the Industrial Workers of the World and as a result many of its members were first and second generation immigrants. Several immigrants such as Mary 'Mother' Jones, Hubert Harrison, Carlo Tresca, Arturo Giovannitti and Joe Haaglund Hill became leaders of the organization.",
        "Chinese Immigration and the Chinese Exclusion Acts. In the 1850s, Chinese workers migrated to the United States, first to work in the gold mines, but also to take agricultural jobs, and factory work, especially in the garment industry.",
        "The Rise of Industrial America, 1877-1900. When in 1873 Mark Twain and Charles Dudley Warner entitled their co-authored novel The Gilded Age, they gave the late nineteenth century its popular name. The term reflected the combination of outward wealth and dazzle with inner corruption and poverty.",
        "American objections to Chinese immigration took many forms, and generally stemmed from economic and cultural tensions, as well as ethnic discrimination. Most Chinese laborers who came to the United States did so in order to send money back to China to support their families there.",
        "The rise of industrial America, the dominance of wage labor, and the growth of cities represented perhaps the greatest changes of the period. Few Americans at the end of the Civil War had anticipated the rapid rise of American industry.",
        "The resulting Angell Treaty permitted the United States to restrict, but not completely prohibit, Chinese immigration. In 1882, Congress passed the Chinese Exclusion Act, which, per the terms of the Angell Treaty, suspended the immigration of Chinese laborers (skilled or unskilled) for a period of 10 years.",
        "Industrial Workers of the World. In 1905 representatives of 43 groups who opposed the policies of American Federation of Labour, formed the radical labour organisation, the Industrial Workers of the World (IWW). The IWW's goal was to promote worker solidarity in the revolutionary struggle to overthrow the employing class.",
        "The railroads powered the industrial economy. They consumed the majority of iron and steel produced in the United States before 1890. As late as 1882, steel rails accounted for 90 percent of the steel production in the United States. They were the nationâs largest consumer of lumber and a major consumer of coal.",
        "This finally resulted in legislation that aimed to limit future immigration of Chinese workers to the United States, and threatened to sour diplomatic relations between the United States and China.",
        "Costa Rica is known as a prime Eco-tourism destination so visitors are assured of majestic views, amazing destination spots and a temperate climate. These factors assure medical tourists of an excellent vacation experience that is conducive for recovery and relaxation.",
        "Medical Tours Costa Rica: Medical Tourism Made Easy! âNo Other Firm Has Helped More Patients. Receive Care Over the Last 15 Yearsâ",
        "Medical Tours Costa Rica difference: At MTCR, our aim is to become your âone-stop shopâ for health care services, so we have put together packages with you, the medical tourist, in mind, offering a wide variety of specialties.",
        "Cost of Medical Treatment in Costa Rica. The following are cost comparisons between Medical procedures in Costa Rica and equivalent procedures in the United States: [sources: 1,2]",
        "Common Treatments done by Medical Tourists in Costa Rica. Known initially for its excellent dental surgery services, medical tourism in Costa Rica has spread to a variety of other medical procedures, including: General and cosmetic dentistry; Cosmetic surgery; Aesthetic procedures (botox, skin resurfacing etc) Bariatric and Laparoscopic surgery",
        "Medical Tours costa Rica office remains within the hospital and the Cook brothers 15 year relationship running the hospitalâs insurance office and seven years running the international patient department serves you the client very well.",
        "About us. Medical Tours Costa Rica has helped thousands of patients and are the innovators in medical travel to Costa Rica. Brad and Bill Cook are visionaries that saw the writing on the wall while running the International insurance office for Costa Ricaâs busiest and most respected hospital The Clinica Biblica.",
        "In an era of rising health care costs and decreased medical coverage, the concept of combining surgery with travel has taken off. The last decade has seen a boom in the health tourism sector in Costa Rica, especially in the area of plastic surgery.",
        "The World Bank ranked Costa Rica as having the highest life expectancy, at 78.7 years. This figure is the highest amongst all countries in Latin America, and is equivalent to the level in Canada and higher than the United States by a year. Top Hospitals for Medical Tourism in Costa Rica",
        "Over the last decade, Costa Rica has evolved from being a mere eco-tourism destination and emerged as a country of choice for foreigners, particularly from United States and Canada. These seek quality healthcare services and surgeries at a much lower price than their home countries.",
        "Colorâurine can be a variety of colors, most often shades of yellow, from very pale or colorless to very dark or amber. Unusual or abnormal urine colors can be the result of a disease process, several medications (e.g., multivitamins can turn urine bright yellow), or the result of eating certain foods."
    ]

    # Process all sentences in one batch to get all routing decisions
    embeddings = routed_model.encode(test_sentences, show_progress_bar=False, batch_size=len(test_sentences))
    routing = routed_model[2].last_routing_decisions
    print(f"   ✓ Test encoding successful")
    print(f"   ✓ Output shape: {embeddings.shape}")
    print(f"   ✓ Teacher usage: {routing.float().mean().item() * 100:.1f}%")
    print(f"   ✓ Student usage: {(1 - routing.float().mean().item()) * 100:.1f}%")

    # Show sample routing decisions with text length
    print(f"\n   Sample routing decisions:")
    for i in range(min(10, len(routing))):
        model_used = "Teacher" if routing[i].item() == 1 else "Student"
        text_preview = test_sentences[i][:60] + "..." if len(test_sentences[i]) > 60 else test_sentences[i]
        print(f"      [{i}] {model_used:8s} (len={len(test_sentences[i]):4d}): {text_preview}")

    # Step 7: Save the model
    print(f"\n7. Saving model...")
    os.makedirs(output_path, exist_ok=True)
    routed_model.save(output_path)
    print(f"   ✓ Model saved to: {output_path}")

    # Step 8: Save metadata
    print(f"\n8. Saving metadata...")
    metadata = {
        "model_type": "routed-sentence-transformer",
        "student_model": student_name_or_path,
        "teacher_model": teacher_name_or_path,
        "threshold": threshold,
        "student_dim": student_dim,
        "teacher_dim": teacher_dim,
        "router_hidden_dim": router_hidden_dim,
        "description": "SentenceTransformer model with dynamic routing between student and teacher",
        "usage": {
            "load": "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('{}')".format(output_path),
            "encode": "embeddings = model.encode(['your', 'texts'])",
            "routing_decisions": "routing = model[2].last_routing_decisions",
            "adjust_threshold": "model[2].threshold = 0.7"
        }
    }

    with open(os.path.join(output_path, 'model_card.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   ✓ Metadata saved to model_card.json")

    # Step 9: Verify it can be loaded with standard API
    print(f"\n9. Verifying standard SentenceTransformer loading...")
    test_model = SentenceTransformer(output_path, device='cpu')
    test_embeddings = test_model.encode(["Test"], show_progress_bar=False)
    assert test_embeddings.shape[1] == student_dim, "Dimension mismatch after reload"
    print(f"   ✓ Model can be loaded with SentenceTransformer('{output_path}')")
    print(f"   ✓ Reload test passed")

    # Print summary
    print("\n" + "=" * 80)
    print("✓ Routed Model Created Successfully!")
    print("=" * 80)
    print(f"\nModel saved to: {output_path}")
    print(f"Embedding dimension: {student_dim}d")
    print(f"Routing threshold: {threshold}")
    print(f"\n📖 Usage:")
    print(f"\n  # Load with standard SentenceTransformer API")
    print(f"  from sentence_transformers import SentenceTransformer")
    print(f"  model = SentenceTransformer('{output_path}')")
    print(f"\n  # Encode (routing happens automatically)")
    print(f"  embeddings = model.encode(['query1', 'query2'])")
    print(f"\n  # Check routing decisions")
    print(f"  routing = model[2].last_routing_decisions")
    print(f"  print(f'Teacher usage: {{routing.float().mean()*100:.1f}}%')")
    print(f"\n  # Adjust threshold")
    print(f"  model[2].threshold = 0.7  # Higher = less teacher usage")
    print(f"  embeddings = model.encode(['query'])")
    print(f"\n" + "=" * 80)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Create a routed retrieval model loadable with SentenceTransformer"
    )
    parser.add_argument(
        "student_name_or_path",
        type=str,
        help="Path or name of student model (e.g., 'artifacts/distilled-768d-mrl-mpnet')"
    )
    parser.add_argument(
        "teacher_name_or_path",
        type=str,
        help="Path or name of teacher model (e.g., 'infly/inf-retriever-v1-pro')"
    )
    parser.add_argument(
        "threshold",
        type=float,
        help="Routing threshold (0-1). If difficulty > threshold, use teacher. Typical: 0.5"
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Directory to save the routed model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Auto-detect if not specified"
    )

    args = parser.parse_args()

    # Validate threshold
    if not 0 <= args.threshold <= 1:
        raise ValueError(f"Threshold must be between 0 and 1, got {args.threshold}")

    # Create routed model
    create_routed_model(
        student_name_or_path=args.student_name_or_path,
        teacher_name_or_path=args.teacher_name_or_path,
        threshold=args.threshold,
        output_path=args.output_path,
        device=args.device
    )


if __name__ == "__main__":
    main()
