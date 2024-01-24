package youtube.data;

import youtube.api.model.Channel;
import youtube.api.model.SearchResultSnippet;
import youtube.api.model.Thumbnail;
import youtube.api.model.ThumbnailDetails;
import youtube.api.util.RepositoryUtils;

import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.*;

public class ChannelData {

    public static List<Channel> getChannelData() throws ParseException {
        List<Channel> result = new ArrayList<>();

        SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'");
        dateFormat.setTimeZone(TimeZone.getTimeZone("UTC"));

        //##### Channel 1 #####

        SearchResultSnippet snippetC1 = new SearchResultSnippet();
        snippetC1.setChannelId("UCxwvutsrqponmlkjihgfedc");
        snippetC1.setChannelTitle("My channel");
        snippetC1.setTitle("My channel");
        snippetC1.setDescription("Welcome to my channel.");
        snippetC1.setLiveBroadcastContent(SearchResultSnippet.LiveBroadcastContent.LIVE);
        snippetC1.setPublishedAt(dateFormat.parse("2020-03-28T10:35:00Z"));
        snippetC1.setThumbnails(generateThumbnailDetails());

        Channel c1 = new Channel("UCxwvutsrqponmlkjihgfedc", snippetC1, "other", "en", Arrays.asList("/m/01k8wb", "/m/05qt0", "/m/02jjt", "/m/0bzvm2"), 6502L,
                2L, "contentOwner1");

        //##### Channel 2 #####

        SearchResultSnippet snippetC2 = new SearchResultSnippet();
        snippetC2.setChannelId("UC0DZmkupLYwc0yDsfocLh0A");
        snippetC2.setChannelTitle("Jelly");
        snippetC2.setTitle("Jelly");
        snippetC2.setDescription("Hi there, welcome to my channel! My name is Jelly and I'm a child friendly Youtuber! Make sure to SUBSCRIBE for daily videos on various games! And ofcourse endless funny moments! \uD83D\uDE02");
        snippetC2.setLiveBroadcastContent(SearchResultSnippet.LiveBroadcastContent.NONE);
        snippetC2.setPublishedAt(dateFormat.parse("2014-05-27T13:19:08Z"));
        snippetC2.setThumbnails(generateThumbnailDetails());

        Channel c2 = new Channel("UC0DZmkupLYwc0yDsfocLh0A", snippetC2, "other", "en", null, 8449503561L, 4084L,
                "contentOwnerJelly");

        //##### Channel 3 #####

        SearchResultSnippet snippetC3 = new SearchResultSnippet();
        snippetC3.setChannelId("UCRpjHHu8ivVWs73uxHlWwFA");
        snippetC3.setChannelTitle("Eurovision Song Contest");
        snippetC3.setTitle("Eurovision Song Contest");
        snippetC3.setDescription("Welcome to the official Eurovision Song Contest channel on YouTube!");
        snippetC3.setLiveBroadcastContent(SearchResultSnippet.LiveBroadcastContent.NONE);
        snippetC3.setPublishedAt(dateFormat.parse("2006-03-05T11:56:13Z"));
        snippetC3.setThumbnails(generateThumbnailDetails());

        Channel c3 = new Channel("UCRpjHHu8ivVWs73uxHlWwFA", snippetC3, "other", "en", Arrays.asList("/m/04rlf", "/m/064t9"), 4364109572L,
                5493L, "contentOwnerESC");

        //##### Channel 4 #####

        SearchResultSnippet snippetC4 = new SearchResultSnippet();
        snippetC4.setChannelId("UCGGhM6XCSJFQ6DTRffnKRIw");
        snippetC4.setChannelTitle("Muse");
        snippetC4.setTitle("Muse");
        snippetC4.setDescription("Origin of Muse out now.  The deluxe box set features 113 tracks in total, with more than 40 unreleased.");
        snippetC4.setLiveBroadcastContent(SearchResultSnippet.LiveBroadcastContent.NONE);
        snippetC4.setPublishedAt(dateFormat.parse("2006-09-28T14:47:54Z"));
        snippetC4.setThumbnails(generateThumbnailDetails());

        Channel c4 = new Channel("UCGGhM6XCSJFQ6DTRffnKRIw", snippetC4, "other", "en", Arrays.asList("/m/04rlf", "/m/06by7", "/m/064t9"), 1796433197L,
                162L, "contentOwnerMuse");

        //##### Channel 5 #####

        SearchResultSnippet snippetC5 = new SearchResultSnippet();
        snippetC5.setChannelId("UCeB7JZwcar2fVoK2w2f9OwA");
        snippetC5.setChannelTitle("Real Betis Balompié");
        snippetC5.setTitle("Real Betis Balompié");
        snippetC5.setDescription("Canal de YouTube oficial del Real Betis Balompié. Visualiza nuestras últimas novedades con RBB Play.\\n--\\nReal Betis Balompié Official YouTube Channel. Watch the latest content with RBB Play.");
        snippetC5.setLiveBroadcastContent(SearchResultSnippet.LiveBroadcastContent.NONE);
        snippetC5.setPublishedAt(dateFormat.parse("2011-05-19T13:15:47Z"));
        snippetC5.setThumbnails(generateThumbnailDetails());

        Channel c5 = new Channel("UCeB7JZwcar2fVoK2w2f9OwA", snippetC5, "other", "es", Arrays.asList("/m/02vx4", "/m/06ntj"), 118320090L,
                2385L, "contentOwnerRBB");

        //##### Channel 6 #####

        SearchResultSnippet snippetC6 = new SearchResultSnippet();
        snippetC6.setChannelId("UCpAyEKkMwTiq8FQ_Jqr4Ckg");
        snippetC6.setChannelTitle("Alex Gasan");
        snippetC6.setTitle("Alex Gasan");
        snippetC6.setDescription("");
        snippetC6.setLiveBroadcastContent(SearchResultSnippet.LiveBroadcastContent.NONE);
        snippetC6.setPublishedAt(dateFormat.parse("2010-10-02T10:43:45Z"));
        snippetC6.setThumbnails(generateThumbnailDetails());

        Channel c6 = new Channel("UCpAyEKkMwTiq8FQ_Jqr4Ckg", snippetC6, "other", "es", Arrays.asList("/m/0410tth", "/m/06ntj", "/m/019_rr", "/m/07yv9", "/m/0bzvm2", "/m/01sjng"),
                265225L, 400L, "contentOwnerAG");

        //##### Channel 7 #####

        SearchResultSnippet snippetC7 = new SearchResultSnippet();
        snippetC7.setChannelId("UCbdSYaPD-lr1kW27UJuk8Pw");
        snippetC7.setChannelTitle("QuantumFracture");
        snippetC7.setTitle("QuantumFracture");
        snippetC7.setDescription("¡Ciencia! ¡y con animaciones! El lado más loco (y real) del Universo... cada jueves.\nEscríbeme: contacto@quantumfracture.es\nSíguenos en Twitter: https://twitter.com/QuantumFracture\ny en Facebook: https://www.facebook.com/QuantumFracture");
        snippetC7.setLiveBroadcastContent(SearchResultSnippet.LiveBroadcastContent.NONE);
        snippetC7.setPublishedAt(dateFormat.parse("2012-08-22T14:53:42Z"));
        snippetC7.setThumbnails(generateThumbnailDetails());

        Channel c7 = new Channel("UCbdSYaPD-lr1kW27UJuk8Pw", snippetC7, "other", "es", Collections.singletonList("/m/01k8wb"), 133103115L, 146L,
                "contentOwnerQF");

        //##### Channel 8 #####

        SearchResultSnippet snippetC8 = new SearchResultSnippet();
        snippetC8.setChannelId("UCnGjxxnmYTUxRl85hrjtEBg");
        snippetC8.setChannelTitle("DonOmarVEVO");
        snippetC8.setTitle("DonOmarVEVO");
        snippetC8.setDescription("Don Omar en Vevo – Videoclips oficiales, Lyric videos, Sesiones en vivo, Entrevistas y muchos más...");
        snippetC8.setLiveBroadcastContent(SearchResultSnippet.LiveBroadcastContent.NONE);
        snippetC8.setPublishedAt(dateFormat.parse("2009-05-12T05:27:47Z"));
        snippetC8.setThumbnails(generateThumbnailDetails());

        Channel c8 = new Channel("UCnGjxxnmYTUxRl85hrjtEBg", snippetC8, "other", "es", Arrays.asList("/m/04rlf", "/m/0g293"), 4168214989L, 48L,
                "contentOwnerDonOmar");

        //##### Channel 9 #####

        SearchResultSnippet snippetC9 = new SearchResultSnippet();
        snippetC9.setChannelId("UC2b0o80VP4cbfPE_v8LQEOA");
        snippetC9.setChannelTitle("Movie Juice");
        snippetC9.setTitle("Movie Juice");
        snippetC9.setDescription("Movie Juice and Premiere are your wrap up of movie and celebrity news from the hottest red carpets here and around the globe plus awesome celebrity star bios and movie trailers.");
        snippetC9.setLiveBroadcastContent(SearchResultSnippet.LiveBroadcastContent.NONE);
        snippetC9.setPublishedAt(dateFormat.parse("2010-10-22T06:31:53Z"));
        snippetC9.setThumbnails(generateThumbnailDetails());

        Channel c9 = new Channel("UC2b0o80VP4cbfPE_v8LQEOA", snippetC9, "show", "en", Arrays.asList( "/m/02vxn", "/m/02jjt"), 14355388L, 796L,
                "contentOwnerMJ");

        //##### Channel 10 #####

        SearchResultSnippet snippetC10 = new SearchResultSnippet();
        snippetC10.setChannelId("UCK8sQmJBp8GCxrOtXWBpyEA");
        snippetC10.setChannelTitle("Google");
        snippetC10.setTitle("Google");
        snippetC10.setDescription("Experience the world of Google on our official YouTube channel. Watch videos about our products,  technology, company happenings and more. Subscribe to get updates from all your favorite Google products and teams.");
        snippetC10.setLiveBroadcastContent(SearchResultSnippet.LiveBroadcastContent.NONE);
        snippetC10.setPublishedAt(dateFormat.parse("2005-09-18T22:37:10Z"));
        snippetC10.setThumbnails(generateThumbnailDetails());

        Channel c10 = new Channel("UCK8sQmJBp8GCxrOtXWBpyEA", snippetC10, "other", "en", null, 2659013380L, 2319L,
                "contentOwnerGoogle");


        result.add(c1);
        result.add(c2);
        result.add(c3);
        result.add(c4);
        result.add(c5);
        result.add(c6);
        result.add(c7);
        result.add(c8);
        result.add(c9);
        result.add(c10);

        return result;
    }

    private static ThumbnailDetails generateThumbnailDetails() {
        ThumbnailDetails details = new ThumbnailDetails();
        String urlFirstPart = "https://yt3.ggpht.com/" + RepositoryUtils.generateEtag().substring(0,12);
        String urlSecondPart = "/AAAAAAAAAAI/AAAAAAAAAAA/" + RepositoryUtils.generateEtag().substring(0,11);
        Thumbnail defaultThumbnail = new Thumbnail();
        defaultThumbnail.setUrl(urlFirstPart + urlSecondPart + "/s88-c-k-no-mo-rj-c0xffffff/photo.jpg");
        Thumbnail medium = new Thumbnail();
        medium.setUrl(urlFirstPart + urlSecondPart + "/s240-c-k-no-mo-rj-c0xffffff/photo.jpg");
        Thumbnail high = new Thumbnail();
        high.setUrl(urlFirstPart + urlSecondPart + "/s800-c-k-no-mo-rj-c0xffffff/photo.jpg");
        details.setDefaultThumbnail(defaultThumbnail);
        details.setMedium(medium);
        details.setHigh(high);
        return details;
    }
}
