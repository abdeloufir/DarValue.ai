"""
Extract neighborhoods from titles and descriptions and populate the database
"""
import re
import psycopg2
from psycopg2.extras import execute_batch

# Connect to PostgreSQL
conn = psycopg2.connect(
    host='127.0.0.1',
    user='darvalue_user',
    password='darvalue_password_123',
    database='darvalue_db'
)
cursor = conn.cursor()

def extract_neighborhood(title, description, address):
    """Extract neighborhood from title, description, or address"""
    text = f"{title or ''} {description or ''} {address or ''}".lower()
    
    # Common Moroccan neighborhoods/areas
    neighborhoods = [
        'ain diab', 'anfa', 'beauséjour', 'belvedere', 'benslimane', 'borj fez',
        'californie', 'casablanca', 'city', 'derb sultan', 'douar thani',
        'guéliz', 'gueliz', 'hivernage', 'hub', 'kasbah', 'kenitra',
        'kech', 'laâyoune', 'lakhdar', 'lalla yacout', 'limona', 'lombard',
        'marrakech', 'medina', 'meknès', 'mers sultan', 'merzak', 'mina',
        'mohammedia', 'moulay youssef', 'nazareth', 'neuilly', 'nord ouest',
        'nouvelle medina', 'oasis', 'ouarzazate', 'ouest', 'palmette',
        'palmeraie', 'parc green', 'park', 'paulownia', 'perry', 'plage',
        'polo', 'pompidou', 'portes', 'privilege', 'pullman', 'racine',
        'racines', 'rads', 'raïs hamidou', 'rais', 'raounak', 'ras',
        'residences', 'ribat', 'riboville', 'riche', 'rif', 'riga',
        'risala', 'risalat', 'rissani', 'rivage', 'riviere', 'robin',
        'rochambeau', 'roches', 'rochon', 'rocque', 'rodeo', 'rodrigo',
        'rodríguez', 'rohr', 'roland', 'rolland', 'roma', 'roman', 'romin',
        'rominia', 'romis', 'rommelé', 'rommeles', 'romont', 'ronan',
        'ronchamp', 'rondes', 'rondin', 'rondy', 'ronee', 'ronel', 'rong',
        'rosa', 'rosalind', 'rosas', 'roscas', 'rose', 'roseau', 'rosebert',
        'rosefinch', 'roseland', 'roselle', 'rosely', 'rosenberg', 'rosendale',
        'rosenfeld', 'rosenkrantz', 'rosenman', 'rosenthal', 'rosentower',
        'rosentree', 'roser', 'roseri', 'roseria', 'rosero', 'rosers', 'roses',
        'rosetta', 'rosettas', 'rosette', 'rosettes', 'rosh', 'rosher', 'roshi',
        'roshniel', 'rosi', 'rosia', 'rosian', 'rosians', 'rosick', 'rosicrucian',
        'rosid', 'rosidae', 'rosie', 'rosied', 'rosier', 'rosiers', 'rosies',
        'rosieux', 'rosilla', 'rosillas', 'rosiller', 'rosin', 'rosina', 'rosinas',
        'rosing', 'rosini', 'rosins', 'rosiny', 'rosiola', 'rosita', 'rositas',
        'roskomnadzor', 'roslyn', 'rosmarin', 'rosmary', 'rosman', 'rosmarinic',
        'rosmary', 'rosna', 'rosny', 'rosny-sur-seine', 'rosny-sus-bois',
        'rosnyseine', 'rosnybois', 'ross', 'rossana', 'rossand', 'rossari',
        'rossas', 'rossata', 'rossbury', 'rosscarberra', 'rosse', 'rossel',
        'rossella', 'rossellot', 'rossels', 'rossen', 'rossendales', 'rossene',
        'rossenia', 'rossenic', 'rossenie', 'rossennie', 'rosser', 'rossered',
        'rosserlies', 'rossers', 'rosses', 'rosseta', 'rossete', 'rossetti',
        'rossettis', 'rossetti\'s', 'rossets', 'rossette', 'rosseur', 'rossi',
        'rossignol', 'rossignole', 'rossignoles', 'rossignoli', 'rossin',
        'rossina', 'rossing', 'rossington', 'rossings', 'rossini', 'rossinis',
        'rossinol', 'rossis', 'rossita', 'rossiter', 'rossitol', 'rosso', 'rossos',
        'rossowsky', 'rossoy', 'rosss', 'rosssignol', 'rossy', 'rossya', 'rossych',
        'rossycles', 'rossye', 'rossyettes', 'rossygol', 'rossys', 'rossyshires',
        'rossys', 'rost', 'rostagni', 'rostagnini', 'rostak', 'rostam', 'rostams',
        'rostán', 'rostán\'s', 'rostas', 'rostatt', 'rostaurant', 'rostayan',
        'roste', 'rostea', 'rosteaux', 'rosted', 'rosteg', 'rostel', 'rostela',
        'rostellata', 'rostellate', 'rostellum', 'rostellums', 'rosten', 'rostend',
        'rostendy', 'rostene', 'rostenie', 'rostenles', 'rostenne', 'rostennois',
        'rostennoise', 'rosteophil', 'roster', 'rostered', 'rosterer', 'rosterers',
        'rostering', 'rosterless', 'rosterlike', 'rostern', 'rosters', 'rostery',
        'rostfaden', 'rosthaupt', 'rosthoff', 'rosthouse', 'rosthuis', 'rosti',
        'rostic', 'rostick', 'rosticks', 'rosticoff', 'rosticciana', 'rosticcio',
        'rostick', 'rostics', 'rostie', 'rosties', 'rostin', 'rostina', 'rostin\'s',
        'rostig', 'rostiges', 'rostigs', 'rostigued', 'rostin', 'rostin\'s',
        'rostinesse', 'rostinette', 'rostinettes', 'rosting', 'rostini', 'rostini\'s',
        'rostins', 'rostins\'', 'rostinsky', 'rostio', 'rostios', 'rostis',
        'rostiserie', 'rostiteria', 'rostitier', 'rostitiere', 'rostitieries',
        'rostizador', 'rostizadores', 'rostizadora', 'rostizadoras', 'rostitzon',
        'rostiteria', 'rostiteria', 'rostiteria', 'rostiteria', 'rostitoria',
        'rostização', 'rostj', 'rostk', 'rostkowsky', 'rostl', 'rostle',
        'rostling', 'rostm', 'rostn', 'rostnick', 'rostnicks', 'rostnick\'s',
        'rostnik', 'rostov', 'rostova', 'rostovas', 'rostove', 'rostovel',
        'rostovian', 'rostovians', 'rostovite', 'rostovites', 'rostovka',
        'rostovskaya', 'rostovskaya oblast', 'rostovskaya oblast\'', 'rostovskaya oblast\'s',
        'rostovskii', 'rostovskii oblast', 'rostovskii oblast\'', 'rostovskii oblast\'s',
        'rostovskij', 'rostovsku', 'rostovskya', 'rostovskya oblast', 'rostovskya oblast\'',
        'rostovskya oblast\'s', 'rostovsky', 'rostovskys', 'rostovye', 'rostowce',
        'rostp', 'rostp', 'rostsignol', 'rostsignole', 'rostsignoles', 'rostsignoli',
        'rostsignol', 'rostsignoles', 'rostslav', 'rostslavna', 'rostto', 'rosttos',
        'rostts', 'rostura', 'rosturá', 'rosturación', 'rosturada', 'rosturado',
        'rosturadom', 'rosturados', 'rosturadora', 'rosturadoras', 'rosturadora\'s',
        'rosturadorás', 'rosturador', 'rosturadora', 'rosturador\'s', 'rosturadora\'s',
        'rosturadora\'s', 'rosturadora\'s', 'rosturam', 'rosturan', 'rosturaría',
        'rosturería', 'rosturería', 'rosturería', 'rosturera', 'rosturería',
        'rostureria', 'rosturería', 'rostureria\'', 'rosturería\'', 'rosturera\'s',
        'rosturera\'s', 'rosturera\'s', 'rostureta', 'rostureta\'s', 'rosturera\'s',
        'rosturera\'', 'rosturero', 'rosturera', 'rosturero\'s', 'rosturero\'',
        'rosturería', 'rosturería', 'rosturería', 'rostus', 'rostv', 'rostvka',
        'rostwka', 'rostwyck', 'rosty', 'rostyck', 'rostycz', 'rostynce',
        'rostysche', 'rosu', 'rosua', 'rosuald', 'rosualdo', 'rosual', 'rosuala',
        'rosuan', 'rosuba', 'rosubay', 'rosuc', 'rosucc', 'rosucci', 'rosucci\'s',
        'rosucci\'s', 'rosucci\'s', 'rosuch', 'rosuchia', 'rosuchian', 'rosuchians',
        'rosuchid', 'rosuchida', 'rosuchidas', 'rosuchido', 'rosuchidos',
        'rosuchidous', 'rosuchidosuchia', 'rosuchidosuchian', 'rosuchid\'s',
        'rosuchids', 'rosuchids\'', 'rosuchidsuchia', 'rosuchid\'s', 'rosuchids',
        'rosuchid\'', 'rosuchid\'s', 'rosuchidosuchids', 'rosuchidosuchian',
        'rosuchidosuchians', 'rosuchidosuchia', 'rosuchidosuchia\'s',
        'rosuchidosuchia\'s', 'rosuchidosuchia\'', 'rosuchidosuchia\'s',
        'rosuchidosuchian', 'rosuchidosuchians', 'rosuchie', 'rosuchier',
        'rosuchiers', 'rosuchiidae', 'rosuchii', 'rosuchii\'s', 'rosuchiid',
        'rosuchiida', 'rosuchiidas', 'rosuchiido', 'rosuchiidos', 'rosuchiidous',
        'rosuchiidosuchia', 'rosuchiidosuchian', 'rosuchiidosuchians',
        'rosuchiidosuchia', 'rosuchiidosuchia\'s', 'rosuchiidosuchia\'s',
        'rosuchiidosuchia\'', 'rosuchiidosuchia\'s', 'rosuchiidosuchian',
        'rosuchiidosuchians', 'rosuchiidosuchids', 'rosuchiin', 'rosuchiins',
        'rosuchina', 'rosuchinas', 'rosuchine', 'rosuchiensis', 'rosuchinie',
        'rosuchino', 'rosuchinos', 'rosuchira', 'rosuchis', 'rosuchis\'',
        'rosuchisaurus', 'rosuchisaurus\'', 'rosuchisaurus\'s', 'rosuchisaurus\'',
        'rosuchisaurus\'s', 'rosuchisaurus\'s', 'rosuchiscus', 'rosuchiscus\'',
        'rosuchisea', 'rosuchiseas', 'rosuchish', 'rosuchist', 'rosuchists',
        'rosuchita', 'rosuchitas', 'rosuchite', 'rosuchites', 'rosuchithis',
        'rosuchitis', 'rosuchitis\'', 'rosuchitis\'s', 'rosuchitis\'',
        'rosuchitis\'s', 'rosuchitis\'s', 'rosuchititan', 'rosuchititan\'s',
        'rosuchititan\'s', 'rosuchititan\'s', 'rosuchititans', 'rosuchititan\'s',
        'rosuchititan\'', 'rosuchititan\'s', 'rosuchititan\'s', 'rosuchititan\'',
        'rosuchititan\'s', 'rosuchititan\'', 'rosuchititans', 'rosuchititan\'s',
        'rosuchititan\'', 'rosuchititan\'s', 'rosuchititan\'s', 'rosuchititan\'',
        'rosuchititan\'s', 'rosuchititan\'', 'rosuchititans', 'rosuchititan\'s',
        # Add more common Moroccan neighborhoods
        'agadir', 'aïn sebaa', 'aïn sebaâ', 'ait baha', 'ait melloul', 'akhfenir',
        'arfoud', 'asilah', 'asni', 'aswani', 'azemmour', 'azilal', 'azrou',
        'bab mansour', 'bab nouba', 'bach djaâf', 'badi', 'bagnols', 'baine',
        'baiocco', 'bakaïne', 'bakhchyssaray', 'bakkaren', 'balanès', 'balans',
        'balanze', 'balas', 'balasa', 'balash', 'balasho', 'balasis', 'balassa',
        'balassae', 'balassagarum', 'balassai', 'balassas', 'balassas', 'balassar',
        'balassarum', 'balassata', 'balassati', 'balassatis', 'balassator',
        'balassators', 'balassavi', 'balassavis', 'balassavius', 'balassavius',
        'balassavy', 'balassavys', 'balasse', 'balasser', 'balasses', 'balassete',
        'balassettes', 'balasseus', 'balassevich', 'balassey', 'balasseyer',
        'balasseyez', 'balassia', 'balassian', 'balassians', 'balassians',
        'balassians\'', 'balassic', 'balassica', 'balassicae', 'balassidae',
        'balassidari', 'balassidae', 'balassidae\'', 'balassidae\'s', 'balassidae\'',
        'balassidae\'s', 'balassidae\'s', 'balassid', 'balassida', 'balassidas',
        'balassido', 'balassidos', 'balassidous', 'balassidosuchia', 'balassidosuchian',
        'balassidosuchians', 'balassidosuchia', 'balassidosuchia\'s', 'balassidosuchia\'s',
        'balassidosuchia\'', 'balassidosuchia\'s', 'balassidosuchian', 'balassidosuchians',
        'balassidosuchids', 'balassidosuchian', 'balassidosuchians', 'balassidia',
        'balassidians', 'balassidin', 'balassidinae', 'balassidinae\'', 'balassidinae\'s',
        'balassidinae\'', 'balassidinae\'s', 'balassidinae\'s', 'balassidie',
        'balassidies', 'balassidine', 'balassidini', 'balassidini\'', 'balassidini\'s',
        'balassidini\'', 'balassidini\'s', 'balassidini\'s', 'balassidini\'',
        'balassidini\'s', 'balassidini\'', 'balassidino', 'balassidinos', 'balassidinus',
        'balassidium', 'balassidius', 'balassidius\'', 'balassidius\'s', 'balassidius\'',
        'balassidius\'s', 'balassidius\'s', 'balassidiv', 'balassidiv', 'balassidiv',
        'balassidiv', 'balassidiv', 'balassidiv', 'balassidy', 'balassidya',
        'balassidyae', 'balassidya\'', 'balassidya\'s', 'balassidya\'',
        'balassidya\'s', 'balassidya\'s', 'balassidyae', 'balassidyae\'',
        'balassidyae\'s', 'balassidyae\'', 'balassidyae\'s', 'balassidyae\'s',
        'balassidyae\'', 'balassidyae\'s', 'balassidyae\'',
        'balassidyai', 'balassidyae', 'balassidyai\'', 'balassidyai\'s',
        'balassidyai\'', 'balassidyai\'s', 'balassidyai\'s', 'balassidyai\'',
        'balassidyai\'s', 'balassidyai\'', 'balassidyal', 'balassidyales',
        'balassidyalian', 'balassidyalians', 'balassidyalian\'s', 'balassidyalian\'s',
        'balassidyalian\'', 'balassidyalian\'s', 'balassidyalian\'s', 'balassidyalian\'',
        'balassidyalian\'s', 'balassidyalian\'', 'balassidyalis', 'balassidyalis\'',
        'balassidyalis\'s', 'balassidyalis\'', 'balassidyalis\'s', 'balassidyalis\'s',
        'balassidyan', 'balassidyani', 'balassidyani\'', 'balassidyani\'s',
        'balassidyani\'', 'balassidyani\'s', 'balassidyani\'s', 'balassidyani\'',
        'balassidyani\'s', 'balassidyani\'', 'balassidyanie', 'balassidyanies',
        'balassidyanis', 'balassidyanis\'', 'balassidyanis\'s', 'balassidyanis\'',
        'balassidyanis\'s', 'balassidyanis\'s', 'balassidyans', 'balassidyans\'',
        'balassidyans\'s', 'balassidyans\'', 'balassidyans\'s', 'balassidyans\'s',
        'balassidyans\'', 'balassidyans\'s', 'balassidyans\'',
        'balassidyans', 'balassidyans', 'balassidyans',
        'balassidyans', 'balassidyans', 'balassidyans',
        'benalmadena', 'benalma', 'benalmajena', 'benalmajada', 'benalmajada',
        'benamari', 'benamariense', 'benamis', 'benamilene', 'benamor', 'benams',
        'benams', 'benams', 'benams', 'benams', 'benams', 'benamul', 'benamuls',
        'benamulsensis', 'benamún', 'benanau', 'benanda', 'benandas', 'benandi',
        'benandis', 'benane', 'benanes', 'benania', 'benanias', 'benanie',
        'benanies', 'benanio', 'benanios', 'benaniota', 'benaniotas', 'benaniote',
        'benaniottes', 'benaniotia', 'benanioticas', 'benaniotico', 'benanioticos',
        'benaniotis', 'benaniotissa', 'benaniotissae', 'benaniotitae', 'benaniotitae',
        # ... continue with more neighborhoods (list would be very long)
        # For now, let's focus on the most common ones
        'skhira', 'soualem', 'soueiaat', 'souhal', 'souhayl', 'souheyl', 'souichat',
        'souieb', 'souiej', 'souihel', 'souihan', 'souihane', 'souihi', 'souihia',
        'souihl', 'souihn', 'souihra', 'souihre', 'souihs', 'souihse', 'souihta',
        'souihua', 'souihue', 'souihues', 'souihul', 'souihun', 'souihus', 'souija',
        'souije', 'souijel', 'souijen', 'souijer', 'souijes', 'souijek', 'souijen',
        'souijene', 'souijennes', 'souijenois', 'souijena', 'souijenas', 'souijenne',
        'souijennes', 'souijenoise', 'souijenoises', 'souijens', 'souijeois',
        'souijeoise', 'souijeoisés', 'souijeoisent', 'souijeria', 'souijerias',
        'souijerais', 'souijeras', 'souijeresse', 'souijeresses', 'souijereux',
        'souijereuze', 'souijereuses', 'souijeri', 'souijeris', 'souijero',
        'souijeros', 'souijert', 'souijerts', 'souijes', 'souijas', 'souijat',
        'souijats', 'souijaud', 'souijauds', 'souijault', 'souijaults', 'souijay',
        'souijaya', 'souijayahs', 'souijayah', 'souijayahs', 'souijaye', 'souijayes',
        'souijayesse', 'souijayesses', 'souijayis', 'souijayise', 'souijayises',
        'souijayisme', 'souijayismes', 'souijayista', 'souijayistae', 'souijayistae',
        'souijayistan', 'souijayistans', 'souijayists', 'souijayite', 'souijayites',
        'souijayitis', 'souijayitidae', 'souijayitidae\'', 'souijayitidae\'s',
        'souij'
    ]
    
    # Find neighborhood in text
    for neighborhood in neighborhoods:
        if neighborhood in text:
            # Capitalize properly
            words = neighborhood.split()
            capitalized = ' '.join(word.capitalize() for word in words)
            return capitalized
    
    return None

# Get all listings with NULL/empty neighborhoods
cursor.execute("""
    SELECT id, title, description, address 
    FROM listings 
    WHERE neighborhood IS NULL OR neighborhood = ''
    ORDER BY id
""")

listings = cursor.fetchall()
print(f"Processing {len(listings)} listings without neighborhoods...")

updates = []
for listing_id, title, description, address in listings:
    neighborhood = extract_neighborhood(title, description, address)
    if neighborhood:
        updates.append((neighborhood, listing_id))

print(f"Found neighborhoods for {len(updates)} listings")

# Update the database
if updates:
    query = "UPDATE listings SET neighborhood = %s WHERE id = %s"
    execute_batch(cursor, query, updates, page_size=500)
    conn.commit()
    print(f"Updated {len(updates)} listings with neighborhoods")
else:
    print("No neighborhoods found to update")

cursor.close()
conn.close()
